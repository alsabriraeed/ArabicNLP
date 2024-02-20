# coding=utf-8


if __name__ == '__main__':
    datasets =['ArabicAC','AlKhaleej','AlArabiya','Akhbarona'] 
    versions =["version2","version3","version4",
				"version5","version6","version7"]
    file = open(r'F:/Arabic dataset prepared/wiki/all_versions/wiki.txt',"w",encoding="utf8")

    for version in versions:
        with open(r'F:/Arabic dataset prepared/wiki/'+version+'/wiki.txt', 'r',encoding="utf8") as f:
            raw_docs = [line for  line in f]
            text =''
            for line in raw_docs:                
#                text =  line.replace("\n","")  
#                text = text + "\n"
                file.write(line)
    file.close()
