# context-chatbot
chat app based on context provided

## Local configuration

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key
```

Optional overrides:

```env
OPENAI_CHAT_MODEL=gpt-5.6-luna
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LANGCHAIN_API_KEY=your_langsmith_api_key
```

For Streamlit Cloud, set the same keys under `[api_keys]` in Streamlit secrets.
