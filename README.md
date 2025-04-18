# PostgreSQL with vector extension

## To creating PostgreSQL Container

At a terminal

1. Pull an Image

```
docker pull pgvector/pgvector:pg17
```

2. Create a Container

```
docker run -e POSTGRES_USER=postgres \
           -e POSTGRES_PASSWORD=Pass.1234 \
           -e POSTGRES_DB=ntdb \
           --name pgvec \
           -p 5432:5432 \
           -d pgvector/pgvector
```

Note: You can map volume at the path `/var/lib/postgresql/data` of the container.


## To creating a PostgreSQL Table

1.  Add the vector extension.

```
CREATE EXTENSION vector;
```

2. Create a table.

```
CREATE TABLE products (
    pid INT PRIMARY KEY,
    pname VARCHAR(100),
    cat VARCHAR(50),
    price DECIMAL(10, 2),
    stock INT,
    spec VARCHAR(300),
    spec_vec vector(1024),
    ads VARCHAR(300),
    ads_vec vector(1024)
);
```
Note: The size of the vector (e.g. 1024) is depended on the vector embeding model. (Our example use "BAAI/bge-reranker-v2-m3".)

## Try Running the Code

*  Learn from `try-pgvec.py`

## Refs
*  https://github.com/pgvector/pgvector



