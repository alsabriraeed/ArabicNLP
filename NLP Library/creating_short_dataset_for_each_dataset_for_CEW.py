# coding=utf-8



class arbic_clean_text:
    

    
    def main(self):
#        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        datasets =['AlKhaleej','AlArabiya','Akhbarona'] #,'AlKhaleej','AlArabiya','Akhbarona'
        versions =["version2.1"]
    #    "version2","version2.1","version3","version3.1",//"version2.1","version3.1",
#        versions =["version2.1"]
#        datasets =['AlKhaleej']#"version2",
        ndocs = 0
    
    #    models = ['cew','pmi','okapibm25','tfrtf','tfrstf']  
    #    models = ['tfrstf']#,'cew'//,'tfrstf'
    #    datasettypes=['short'] # 'long'
        datasettypes = ['long']# ,'short', 'long'
    #    datasettypes = ['long']
        for datasettype in datasettypes:
            for dataset in datasets:
                for version in versions:
                    print(datasettype+" "+dataset +" "+ version+" ok")
                    if dataset =='AlKhaleej':
                        ndocs = 3000
                    elif dataset =='AlArabiya':
                        ndocs  = 3500
                    else:
                        ndocs = 4000
        #            path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\AlArabiya/'
                #    path = r'C:\Users\Raeed\workspace\STTM\STTM\dataset\ArabicCorpusACTwosentence/'
#                    path = 'F:\Arabic_dataset_prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
    #                path = '/content/drive/My Drive/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
    #                path = '/home1/raeed/Arabic dataset prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                    path = '/nuist/scratch/wangwen/wangwen_ali/raeed/Arabic_dataset_prepared/'+dataset+r'/'+version+'/'+datasettype+'/'
                    name = dataset+'c'
                    file = open(path + name +'10.txt',"w",encoding="utf8")
                    filel = open(path + name +'label10.txt',"w",encoding="utf8")
                    with open(path + name + '.txt', 'r',encoding="utf-8-sig") as f,\
                            open(path + name + '_label.txt', 'r',encoding="utf-8-sig") as fl:
                        raw_docs = [line for  line in f]
                        raw_labels = [line for  line in fl]
                    print(len(raw_docs))
                    print(len(raw_labels))
                #    print(s.punc_to_remove)
                    text = ''
                    label = ''
                    count = 0
                    for line,linelabel in zip(raw_docs,raw_labels):
                        text = ''
                        label = ''
                        text =  line.replace("\n","") 
                        label =  linelabel.replace("\n","")
                        if count < ndocs:
                                text = text + "\n"
                                label = label + "\n"
                                file.write(text)
                                filel.write(label)
                        count+=1
                    file.close()
                    filel.close()
if __name__ == '__main__':
    obj = arbic_clean_text()
    obj.main()
