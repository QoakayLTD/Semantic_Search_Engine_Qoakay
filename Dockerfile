FROM python:3.7.10-slim-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements-new.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]
