FROM python:2.7
ADD requirements.txt /
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
ADD . /
EXPOSE 5000
CMD ["/train_and_run.sh", "/data/training.csv"]

