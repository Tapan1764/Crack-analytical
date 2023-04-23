import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="hcd"
)

mycursor = mydb.cursor()

sql = "INSERT INTO crack_data (length, width, depth) VALUES (10, 20, 30)"
mycursor.execute(sql)

mydb.commit()

print(mycursor.rowcount, "record inserted.")