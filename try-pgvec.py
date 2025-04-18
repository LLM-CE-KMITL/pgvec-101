import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from transformers import AutoTokenizer, AutoModel
import torch

#### TABLE SCHEMA #####

'''
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
'''

#### CREATE EMBEDDING MODEL #####
 

em_name = "BAAI/bge-reranker-v2-m3"
em_tokenizer = AutoTokenizer.from_pretrained(em_name)
em_model = AutoModel.from_pretrained(em_name)

def get_embedding(text):
    inputs = em_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = em_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding / np.linalg.norm(embedding)

#### CREATE EXAMPLE DATA #####


data = {
    "pid": range(1, 21),
    "pname": [
        "Laptop", "Smartphone", "Tablet", "Monitor", "Keyboard",
        "Mouse", "Printer", "Desk Lamp", "Webcam", "Headphones",
        "USB Cable", "Charger", "Speaker", "External HDD", "SSD",
        "Graphics Card", "Motherboard", "Power Supply", "RAM", "Cooling Fan"
    ],
    "cat": [
        "Electronics", "Electronics", "Electronics", "Electronics", "Accessories",
        "Accessories", "Electronics", "Home", "Electronics", "Accessories",
        "Accessories", "Accessories", "Accessories", "Storage", "Storage",
        "Components", "Components", "Components", "Components", "Components"
    ],
    "price": [
        999.99, 699.99, 329.99, 199.99, 49.99,
        29.99, 149.99, 39.99, 89.99, 59.99,
        9.99, 19.99, 79.99, 99.99, 119.99,
        299.99, 189.99, 109.99, 79.99, 39.99
    ],
    "stock": [
        50, 75, 60, 40, 150,
        180, 30, 90, 55, 85,
        200, 120, 65, 70, 100,
        35, 45, 40, 100, 110
    ],
    "spec": [
        "High-performance laptop with 16GB RAM and 512GB SSD.",
        "Latest-gen smartphone with AMOLED display and 5G support.",
        "10-inch tablet ideal for media and productivity.",
        "24-inch full HD monitor with vibrant color reproduction.",
        "Ergonomic keyboard with backlit keys for comfort typing.",
        "Wireless optical mouse with precision tracking.",
        "All-in-one color printer with wireless connectivity.",
        "Adjustable LED desk lamp with touch control.",
        "HD webcam perfect for video calls and streaming.",
        "Noise-cancelling over-ear headphones with deep bass.",
        "Durable USB-A to USB-C cable, 1 meter in length.",
        "Fast-charging USB wall charger with dual ports.",
        "Bluetooth speaker with rich sound and compact design.",
        "1TB external hard drive with USB 3.0 support.",
        "512GB solid state drive with fast read/write speeds.",
        "High-end graphics card for gaming and creative work.",
        "ATX motherboard with support for latest CPUs and RAM.",
        "500W power supply with 80+ efficiency rating.",
        "8GB DDR4 RAM stick with heat spreader.",
        "Silent cooling fan for desktop PC cases."
    ],
    "ads": [
        "A portable personal computer for work, study, or entertainment.",
        "A mobile device that combines phone, internet, and app capabilities.",
        "A touchscreen device used for browsing, reading, and media.",
        "A screen that displays video output from a computer or device.",
        "An input device used to type on computers and other devices.",
        "A hand-held device for moving a cursor and clicking items on screen.",
        "A machine that prints documents and images onto paper.",
        "A light source designed for a desk or workstation.",
        "A small camera used for video chatting or online meetings.",
        "Audio device worn on ears to listen to music or sound privately.",
        "A cable that connects devices for power or data transfer.",
        "A device that charges electronic gadgets from a power outlet.",
        "A device that plays music or sound wirelessly via Bluetooth.",
        "Portable storage for backing up and transferring large files.",
        "A fast, compact storage device for quick file access.",
        "A component that handles image rendering for games and media.",
        "The main circuit board that connects all computer components.",
        "Supplies power to all components in a desktop computer.",
        "Temporary memory used by computers to store data for quick access.",
        "Helps cool computer parts and maintain optimal temperature."
    ]
}

df = pd.DataFrame(data)

df

#### ADD VECTOR DATA #####


df['spec_vec'] = df['spec'].apply(lambda x: get_embedding(x).tolist())

df['ads_vec'] = df['ads'].apply(lambda x: get_embedding(x).tolist())


#### INSERT DATA INTO POSTGRES #####

engine = create_engine('postgresql://postgres:Pass.1234@0.0.0.0:5432/ntdb')


df.to_sql(name='products', con=engine, if_exists='append', index=False)

#### TRY VECTOR QUERY #####

query = 'maintain optimal temperature.'
qvec = get_embedding(query).tolist()
res = pd.read_sql("SELECT pid, pname, ads, ads_vec <=> '{0}' as cs FROM products ORDER BY cs LIMIT 5".format(qvec), con=engine)
res
