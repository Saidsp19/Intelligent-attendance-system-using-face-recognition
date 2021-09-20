import pandas as pd
import datetime
import numpy as np

now= datetime.datetime.now()
date = datetime.datetime.now().date()
time = datetime.datetime.now().time()

folder = 'Attendence//'
file_name = folder + "Attendence_{}.xlsx".format(date.strftime("%d-%m-%Y"))

column_names = ["Name" ,"Date" ,"Outtime" ,"Intime"]
A = pd.DataFrame(columns = column_names)

names = np.load('../Attendence system/name_dict.npy',allow_pickle='TRUE').item()
data =pd.read_excel(file_name)
grouped = data.groupby(data.Name)

for i in range(8):
    D = pd.DataFrame()
    try:
        D = grouped.get_group(names[i])
    except KeyError:
        pass    
    if not D.empty:
        It = D.Time.min()
        Ot = D.Time.max()
        A = A.append({"Name" : names[i],"Date" :date.strftime("%d/%m/%Y") ,
                                    "Outtime":Ot ,"Intime":It},ignore_index = True)
    
attendance = A.to_numpy()


np.savetxt(folder + "Attendence_{}.csv".format(date.strftime("%d-%m-%Y")),attendance,fmt = "%s",delimiter = "," ,header ="Name,Date,Outtime,Intime",comments = "")