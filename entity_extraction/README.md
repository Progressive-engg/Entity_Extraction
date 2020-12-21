Entity_Extraction
==============================

A short description of the project.
Note : All the required code is in "src" Folder
Project Organization
------------
      

    ├── requirements.txt   <- All the package & library dependencies are written here `
    │          
    ├── src
    │   │
    │   ├── data 
              - 
    │   │   └── make_dataset.py-->This particular script is extracting data from image and cleaning it 
                                   generating dataset and writting into excel file
                Building_Dataset.log -> This is log file for this scipt
                required2_dataset.xlsx-> This data has been extracted from image using make_dataset.py script and after 
                                         that labelling is done manually 
                Required_Dataset3.csv -> just converted .xlsx file to csv file after labelling is done .
                                         This is final dataset which will be used futher 
                                         

    │   ├── models         I have tried to solve this problem statement using two diff. approach
    
    │   │   ├── Model1_Bert.py --> This script contains Bert approach for solving this problem
                                    From Loading .csv file till model validation .
                model.h5 ----> model weights are stored in this file 
                                     
    │   │   └── Model2_BILSTM.py ---> This script contains  BILSTM approach to solve this problem
    │   │
    │   └── visualization  
    │       └── visualize.py --> This script contains data visualization part to understand data better .
    
    
   Note :
   
   All the coding is in four different file :
   
   1)make_dataset.py --> Building dataset 
   
   2)visualize.py  --->Data visualization part
   
   3)Mode1_Bert.py  ---> Bert approach for solving this problem  (from loading dataset till model validation)
   
   4)Model2_BILSTM.py ---> BILSTM approach for solving this problem
   
   
   requirements.txt ---> All the required dependency mention here 
   
   
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
