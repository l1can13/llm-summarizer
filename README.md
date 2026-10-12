# ЛР №2. Автоматическая суммаризация новостей (CNN/DailyMail)

Курс «Нейронные сети», трек А, вариант 4 «Суммаризация или генерация текста», задача — автоматическая суммаризация.

Сравниваются три подхода на одной тестовой выборке:

| Модель | Подход | Где выполняется |
|---|---|---|
| RoBERTa-base | экстрактивная суммаризация (выбор 3 предложений), дообучение на CNN/DailyMail | локально, GPU |
| Qwen3.6-35B-A3B (GGUF Q4_K_M) | генерация без дообучения | локально, LM Studio |
| GPT-6 Luna | генерация без дообучения | OpenAI API |

Lead-3 (первые три предложения статьи) — точка отсчёта. Метрики: ROUGE-1/2/L, BLEU, BERTScore.

Результаты — в ноутбуке и в `outputs/`.

## Структура

```
lr2/
├── lr2_summarization.ipynb  # эксперимент целиком
├── src/
│   ├── config.py            # пути, модели, параметры
│   ├── data.py              # загрузка, очистка, предложения, разметка «оракула», анализ
│   ├── extractive.py        # экстрактивная модель на RoBERTa: обучение, предсказание, задержка
│   ├── llm.py               # промпты, клиенты LM Studio и OpenAI, кэш ответов
│   ├── metrics.py           # ROUGE, BLEU, BERTScore, доля новых n-грамм, оракул
│   └── viz.py               # графики
├── outputs/                 # таблицы, графики, ответы LLM (generations/*.jsonl)
├── .env                     # OPENAI_API_KEY (в репозиторий не входит)
└── requirements.txt
```

## Данные

[CNN/DailyMail 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail) (Apache-2.0), скачивается первой ячейкой раздела «Данные» в `../datasets/cnn_dailymail/`.

## Запуск

```
python -m venv .venv
.venv\Scripts\pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install -r requirements.txt ipykernel
```

- Ключ OpenAI — в файле `.env`: `OPENAI_API_KEY=...`.
- Qwen — через LM Studio (модель `qwen/qwen3.6-35b-a3b`); ноутбук сам запускает сервер и загружает/выгружает модель командой `lms`.
- Ответы LLM сохраняются построчно в `outputs/generations/*.jsonl`: прерванную ячейку можно запустить снова, готовые ответы не запрашиваются повторно.

## Google Colab

<ссылка на ноутбук в Colab>

В Colab ноутбук открывается с сохранёнными результатами. LM Studio там недоступен, поэтому ответы Qwen берутся из сохранённых файлов.
