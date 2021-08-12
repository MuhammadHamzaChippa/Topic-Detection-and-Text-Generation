FROM python:3.8-slim-buster
RUN apt-get update -y
RUN apt-get install  libsasl2-dev python-dev libldap2-dev libssl-dev libsnmp-dev -y
RUN pip3 install --no-cache --upgrade pip setuptools
WORKDIR /text_gen
COPY . /text_gen
RUN pip install -r requirements.txt                                                                            
CMD [ "python", "text_gen.py" ]
