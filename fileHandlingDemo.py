import matplotlib as mp
import pandas as pd
df = pd.read_csv(r"C:\Users\Admin\Desktop\pandas\test.csv")
print(df)

# unicode error works if we are using a normal string as a path. In that case place an 'r ' before the
#string. This tells the compiler to see the string as a raw string and not any other normal string