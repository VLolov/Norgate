import sqlite3
import pandas as pd

conn = sqlite3.connect('customer.db')       # or ':memory:;

c = conn.cursor()

c.execute("DROP TABLE IF EXISTS customers;")

c.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        custID INTEGER PRIMARY KEY, 
        firstname NVARCHAR(50), 
        lastname NVARCHAR(50), 
        email NVARCHAR(50)
    );
""")

# c.execute("INSERT INTO customers VALUES('Mary', 'Brown', 'mary@codemy.com')")

many_customers = [
    ('Wes', 'Brown', 'wes@codemy.com'),
    ('Steph', 'Kuewa', 'steph@kuewa.com'),
    ('Dan', 'Pas', 'dan@pas.com'),
]

c.executemany("insert into customers values(NULL, ?, ?, ?)", many_customers)

c.execute("select * from customers")
print(c.fetchall())


df = pd.read_sql_query('select * from customers', conn)
print(df)
df_new = df[:-1]
# df_new.reset_index().drop(['index'], axis=1) # remove index completely, but then we still have the primary key
# df_new = df_new.to_sql("customers", conn, if_exists="replace") # inserts one more key in the table ?!?

conn.commit()
c.close()
conn.close()



