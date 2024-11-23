FROM postgres:13

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    postgresql-server-dev-13

RUN cd /tmp && \
    git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

RUN usermod -u 1000 postgres

RUN echo "CREATE EXTENSION vector;" > /docker-entrypoint-initdb.d/create_vector_extension.sql