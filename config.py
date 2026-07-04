from environs import Env

env = Env()
env.read_env()

API_KEY = env.str("API_KEY")
MODEL = env.str("MODEL","claude-sonnet-4-6")
MAX_TOKENS = env.int("MAX_TOKENS",2048)