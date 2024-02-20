# coding=utf-8


if __name__ == '__main__':
    datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona'] 
    versions =["version2","version3","version4",
				"version5","version7"]
    
    for dataset in datasets:
        
        for version in versions:
            
            file = open(r'F:\Arabic dataset prepared/wiki/'+ dataset +'/'+version+'/wiki.txt',"w",encoding="utf8")
#            for version1 in versions:
            for dataset1 in datasets:
                
                if dataset != dataset1:
                    
                    with open(r'F:\Arabic dataset prepared/'+dataset1+'/'+version+'/long/'+dataset1+'c.txt', 'r',encoding="utf8") as f:
                        raw_docs = [line for  line in f]
                        text =''
                    for line in raw_docs:                
        #                text =  line.replace("\n","")  
        #                text = text + "\n"
                        file.write(line)
            file.close()
