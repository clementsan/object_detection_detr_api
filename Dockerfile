FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --timeout=360 --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 8000

# CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
