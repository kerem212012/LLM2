import json
import sys

from anthropic import Anthropic, APIConnectionError, AuthenticationError
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import API_KEY, MAX_TOKENS, MODEL

console = Console()

ANALYSIS_SYSTEM_PROMPT = """You are a text analysis assistant. Return valid JSON only.
Your response must start with { and end with }.

Schema:
{
  "summary": "A concise summary of the text in 1-2 sentences",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "sentiment": {
    "sentiment": "positive|negative|neutral",
    "score": 0.0,
    "reasoning": "Brief explanation of why this sentiment score was assigned"
  },
  "language": "en",
  "word_count": 123
}

Rules:
- summary: write in the same language as the input text
- keywords: extract 3 to 7 most important words or phrases
- sentiment.sentiment: must be exactly 'positive', 'negative', or 'neutral'
- sentiment.score: float from -1.0 (very negative) to 1.0 (very positive), 0.0 is neutral
- sentiment.reasoning: 1-2 sentences explaining the score
- language: ISO 639-1 code of the INPUT text language, not the language of this prompt
- word_count: approximate number of words in the input text

Examples of correct responses:

Input: 'Python is a great programming language for beginners and experts alike.'
Output: {
  "summary": "Python is praised as a versatile language suitable for all skill levels.",
  "keywords": ["Python", "programming language", "beginners", "experts"],
  "sentiment": {"sentiment": "positive", "score": 0.8, "reasoning": "The text expresses clear enthusiasm about Python using positive words like great."},
  "language": "en",
  "word_count": 12
}

Input: 'The software crashed again and lost all my work. I am very frustrated.'
Output: {
  "summary": "The user experienced a software crash that caused data loss and frustration.",
  "keywords": ["software", "crashed", "lost", "work", "frustrated"],
  "sentiment": {"sentiment": "negative", "score": -0.9, "reasoning": "Strong negative emotions expressed through words like crashed, lost, and frustrated."},
  "language": "en",
  "word_count": 15
}

Input: 'The meeting is scheduled for 3pm tomorrow in room 204.'
Output: {
  "summary": "A meeting is announced for 3pm tomorrow in room 204.",
  "keywords": ["meeting", "scheduled", "3pm", "tomorrow", "room 204"],
  "sentiment": {"sentiment": "neutral", "score": 0.0, "reasoning": "The text is purely informational with no emotional content."},
  "language": "en",
  "word_count": 10
}
"""

SAMPLE_TEXTS = [
    "Python — это мощный и удобный язык программирования. Он широко используется для автоматизации, анализа данных и веб-разработки.",
    "This is a sharply critical article about AI, arguing that the industry often trades on hype while producing fragile systems and serious ethical problems.",
    "Der Herbst ist eine wunderschöne Jahreszeit. Die Blätter werden golden und rot, und die Luft wird spürbar kühler.",
]


class SentimentResult(BaseModel):
    sentiment: str
    score: float
    reasoning: str


class TextAnalysis(BaseModel):
    summary: str
    keywords: list[str]
    sentiment: SentimentResult
    language: str
    word_count: int


def calc_cost(usage) -> float:
    input_tokens = getattr(usage, "input_tokens", 0)
    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0)
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0)
    cost = (
        input_tokens * 3
        + cache_creation_tokens * 3.75
        + cache_read_tokens * 0.30
    ) / 1_000_000
    return cost


def print_usage_stats(usage) -> None:
    print(f"Input tokens: {getattr(usage, 'input_tokens', 0)}")
    print(f"Cache creation input tokens: {getattr(usage, 'cache_creation_input_tokens', 0)}")
    print(f"Cache read input tokens: {getattr(usage, 'cache_read_input_tokens', 0)}")
    print(f"Estimated cost: ${calc_cost(usage):.6f}")


def show_analysis(result: TextAnalysis, usage=None) -> None:
    sentiment = result.sentiment.sentiment
    border_style = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "yellow"
    panel = Panel.fit(
        f"[b]Summary[/b]\n{result.summary}\n\n"
        f"[b]Keywords[/b]\n{', '.join(result.keywords)}\n\n"
        f"[b]Sentiment[/b]\n{result.sentiment.sentiment} ({result.sentiment.score})\n"
        f"{result.sentiment.reasoning}\n\n"
        f"[b]Language[/b]\n{result.language}\n\n"
        f"[b]Word count[/b]\n{result.word_count}",
        title="Text analysis",
        border_style=border_style,
    )
    console.print(panel)
    if usage is not None:
        print_usage_stats(usage)


def analyze_text(text: str):
    client = Anthropic(api_key=API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": ANALYSIS_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": text}],
    )

    raw = response.content[0].text.strip()
    try:
        data = json.loads(raw)
        result = TextAnalysis.model_validate(data)
        return result, response.usage
    except json.JSONDecodeError as exc:
        print(f"Не JSON: {exc}")
        print(f"Ответ модели: {raw[:300]}")
        raise
    except ValidationError as exc:
        print(f"Неверная схема: {exc}")
        raise


def batch_analyze(texts: list[str]) -> None:
    results = []
    usages = []

    for index, text in enumerate(texts, 1):
        print(f"Анализирую {index}/{len(texts)}...")
        try:
            result, usage = analyze_text(text)
            results.append(result)
            usages.append(usage)
        except (AuthenticationError, APIConnectionError) as exc:
            console.print(f"[bold red]Ошибка:[/bold red] {exc}")
            continue
        except Exception as exc:
            console.print(f"[bold red]Ошибка:[/bold red] {exc}")
            continue

    table = Table(title="Результаты анализа")
    table.add_column("#")
    table.add_column("Язык")
    table.add_column("Слов")
    table.add_column("Тональность")
    table.add_column("Оценка")
    table.add_column("Ключевые слова")
    table.add_column("Резюме", max_width=60)

    for idx, result in enumerate(results, 1):
        sentiment_style = (
            "green"
            if result.sentiment.sentiment == "positive"
            else "red"
            if result.sentiment.sentiment == "negative"
            else "dim"
        )
        table.add_row(
            str(idx),
            result.language,
            str(result.word_count),
            result.sentiment.sentiment,
            str(result.sentiment.score),
            ", ".join(result.keywords[:4]),
            result.summary,
            style=sentiment_style,
        )

    console.print(table)

    total_cache_read = sum(getattr(usage, "cache_read_input_tokens", 0) for usage in usages)
    total_input_tokens = sum(getattr(usage, "input_tokens", 0) for usage in usages)
    total_cache_creation = sum(getattr(usage, "cache_creation_input_tokens", 0) for usage in usages)
    total_tokens = total_input_tokens + total_cache_read + total_cache_creation

    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total cache creation input tokens: {total_cache_creation}")
    print(f"Total cache read input tokens: {total_cache_read}")
    print(f"Total tokens: {total_tokens}")

    if total_cache_read > 0:
        console.print("[green]Prompt caching is active.[/green]")


def main() -> int:
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        try:
            result, usage = analyze_text(text)
        except AuthenticationError as exc:
            console.print(f"[bold red]Authentication error:[/bold red] {exc}")
            return 1
        except APIConnectionError as exc:
            console.print(f"[bold red]API connection error:[/bold red] {exc}")
            return 1

        show_analysis(result, usage)
        return 0

    try:
        batch_analyze(SAMPLE_TEXTS)
    except AuthenticationError as exc:
        console.print(f"[bold red]Authentication error:[/bold red] {exc}")
        return 1
    except APIConnectionError as exc:
        console.print(f"[bold red]API connection error:[/bold red] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
