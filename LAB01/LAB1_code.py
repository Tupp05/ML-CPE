import pandas as pd

#อ่านไฟล์ csv
df = pd.read_csv(r"C:\Users\tuppo\OneDrive\เดสก์ท็อป\ML Lab\silent_sting_triage_data.csv")

#ดูข้อมูลเบื้องต้น
print("Shape : ", df.shape)
print("Columns : ", df.columns.tolist())
print()

#แสดงข้อมูล 5 แถวแรก
print("----------------------------- Sample --------------------------------")
print()
print(df.head())
print()

#แสดง Data Types
print("---------- Data Types ------------")
print()
print(df.dtypes)
print()

#แสดง Summary Statistics
print("---------------- Summary Statistics -----------------")
print()
print(df.describe())
print()

#แสดง Missing Values 
print("--------- Missing Values ----------")
print()
print(df.isnull().sum())
print()

#แสดง Duplicate Records
print("Duplicate Records : ", df.duplicated().sum())
print()

#แสดง Class Distribution
print("---- Class Distribution ----")
print()
print(df['Bite_Source_Target'].value_counts())
print()
print(df['Muscle_Paralysis_Present'].value_counts()) #1 เป็น 0 ไม่เป็น
print()
print(df['Blood_Coagulation_Failure'].value_counts())
print()

#souce : https://www.kaggle.com/datasets/jacopoferretti/emergency-triage-venomous-bites-dataset
print("Souce : Emergency Triage: Venomous Bites Dataset (https://www.kaggle.com/datasets/jacopoferretti/emergency-triage-venomous-bites-dataset)")