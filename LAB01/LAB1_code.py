import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder

# Part 1 ===================================================

#อ่านไฟล์ csv
df = pd.read_csv(r"C:\Users\tuppo\OneDrive\เดสก์ท็อป\ML Lab\silent_sting_triage_data.csv")

#ดูข้อมูลเบื้องต้น
print("Shape : ", df.shape)
print("Columns : ", df.columns.tolist())
print()

#แสดงข้อมูล 5 แถวแรก
print("----------------- Sample -----------------")
print()
print(df.head())
print()

#แสดง Data Types
print("----------------- Data Types -----------------")
print()
print(df.dtypes)
print()

#แสดง Summary Statistics
print("----------------- Summary Statistics -----------------")
print()
print(df.describe())
print()

#แสดง Missing Values 
print("----------------- Missing Values -----------------")
print()
print(df.isnull().sum())
print()

#แสดง Duplicate Records
print("Duplicate Records : ", df.duplicated().sum())
print()

#แสดง Class Distribution
print("----------------- Class Distribution -----------------")
print()
print(df['Bite_Source_Target'].value_counts())
print()
print(df['Muscle_Paralysis_Present'].value_counts()) #1 เป็น 0 ไม่เป็น
print()
print(df['Blood_Coagulation_Failure'].value_counts())
print()

# Part 2 ===================================================

#แสดง Histogram 
plt.hist(df["Bite_Source_Target"]) #ใช้ข้อมูลจาก Bite_Source_Target
plt.title("Histogram") #ชื่อกราฟ
plt.xlabel("Bite Source Target") #ชื่อแกนX
plt.ylabel("Statistics record") #ชื่อแกนY
plt.show() #แสดงกราฟ

#แสดง Correlation Heatmap  
df_encoded = pd.get_dummies(
    df,
    columns=["Local_Swelling", "Bite_Source_Target"],
    dtype=int
    ) #เอาข้อมูลจากใน Local_Swelling และ Bite_Source_Target มาเก็บไว้ใน df_encoded
corr = df_encoded.corr(numeric_only=True) #ใช้ข้อมูลจาก df_encoded และ ข้อมูลที่เป็นตัวเลข
sb.heatmap(corr, annot=True) #สร้าง Heatmap จากข้อมูลใน corr และใส่เลขความสัมพันธ์ลงไปด้วย
plt.show() #แสดง Correlation Heatmap 

# Part 3 ===================================================

#แสดง Missing Value ที่ถูกแก้แล้ว
print("----------------- Missing Value Handling -----------------")
print()
df["Local_Swelling"] = df["Local_Swelling"].fillna("None") #เติมข้่อมูลในcolumn Local_Swelling ที่หายด้วย None
print(df.isnull().sum())
print()

#เนื่องจากข้อมูลเราไม่มีข้อมูลซ้ำจึงไม่ต้องทำการ Duplicate Removal
# ตัวอย่าง
# print(df.duplicated().sum())
# df = df.drop_duplicates()

#เนื่องจากข้อมูลเราไม่มีข้อมูลที่ผิดจึงไม่ต้องทำการ Incorrect Data Correction
# ตัวอย่าง
# สมมติ Gender มีข้อมูลผิด เช่น M, F แต่ต้องการให้เป็น Male, Female
# df["Gender"] = df["Gender"].replace({
#     "M": "Male",
#     "F": "Female"
# })
# หรือถ้า Local_Swelling มี mild, Mild, MILD ปนกัน
# df["Local_Swelling"] = df["Local_Swelling"].str.capitalize()

#เนื่องจากข้อมูลเราเป็น Data Type ที่ต้องการอยู่แล้วจึงไม่จำเป็นต้อง Data Type Conversion Compare
# ตัวอย่าง
# เช่น ถ้า Age เป็น str แล้วต้องการให้เป็น int
# df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

#แสดง Mean
print("----------------- Mean -----------------")
print()
print(df.select_dtypes(include="number").agg(["mean"]))
print()

#แสดง Median
print("----------------- Median -----------------")
print(df.select_dtypes(include="number").agg(["median"]))
print()

# Part 4 ===================================================

#แสดง Label Encoding  
le = LabelEncoder()
df["Gender_Encoded"] = le.fit_transform(df["Gender"]) #แปลงค่า Male, Female เป็น 1 และ 0
print("----------------- Result of Label Encoding -----------------")
print()
print(df["Gender_Encoded"])
print()

#แสดง One-Hot Encoding 
print("----------------- Result of One-Hot Encoding -----------------")
print()
df_enc_BST= pd.get_dummies(
    df,
    columns=["Bite_Source_Target"]
) #แตกcolumn และแปลงค่าเป็น True, False
print(df_enc_BST)
df_enc_LS = pd.get_dummies(
    df,
    columns=["Local_Swelling"]
)
print(df_enc_LS)
print()

#souce : https://www.kaggle.com/datasets/jacopoferretti/emergency-triage-venomous-bites-dataset
print("Souce : Emergency Triage: Venomous Bites Dataset (https://www.kaggle.com/datasets/jacopoferretti/emergency-triage-venomous-bites-dataset)")
