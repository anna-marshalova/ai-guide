# 🌏🧳🛩️  AI Travel Guide
> Ассистент путешественника, который предоставляет информацию о культурных особенностях, обычаях и достопримечательностях по запросу пользователя о месте.

## Запуск

Для запуска приложения локально выполните следующие шаги:

1. Установите необходимые зависимости:
   ```
   pip install -r requirements.txt
   ```

2. Если запускаете приложение впервые, то скачайте данные для RAG из папки и положите в [папку](https://drive.google.com/drive/folders/1vZmVLdmalDOYs8N7aTzUBaJ6sprVpo5f?usp=sharing) data в корне репозитория. При запуске данные будут проиндексированы и записаны в vectorstore.

3. Приложение использует LLM GigaChat по API, поэтому для коррректной работы необходимо добавить ключ API_KEY в файл .env

4. Запустите Gradio приложение следующей командой:
   ```
   python app.py
   ```

5. После запуска приложение будет доступно на localhost или по публичной ссылке

## Структура репозитория
```
├── src
│   ├── data -- парсинг и предобработка данных
|       ├── data_parsing.py -- парсинг данных с wiki-ресурсов
|       └── data_processing.py -- предобработка данных перед индексацией
|   ├── rag.py -- собственно RAG
|   ├── retriever.py -- Retriever для индексации и создания vectorstore
|   ├── interface.py -- интерфейс на Gradio
|
├── app.py -- основное приложение
└── requirements.txt -- зависимости
```