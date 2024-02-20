import clustering_fun as func

if __name__ == "__main__":
    # tfidf =  func.load_list("tf_idf")
    # tfidf = func.load_list([[1, 2, 3, 4],
    #                         [2, 4, 2, 1]])

    # woorkbook = func.create_excel()
    # woorkbook.create_sheet('Silhouette')
    # sheet1 = woorkbook['Silhouette']
    #
    s_avg = func.clustering([[1, 2, 3, 4],
                             [2, 4  , 2, 1],
                             [3, 1, 3, 1],
                             [2, 1, 3, 1]])
    print(s_avg)
    # func.save_list("s_avg",s_avg)

    # for i in range(len(s_avg)):
    #     sheet1.cell(row=i+1, column=1).value = i+2
    #     sheet1.cell(row=i+1, column=2).value = s_avg[i]

    # tfidf =  func.load_list("tf_idf_2")
    # woorkbook.create_sheet('Silhouette_2')
    # sheet1 = woorkbook['Silhouette_2']
    #
    # s_avg = func.clustering(tfidf)
    # func.save_list("s_avg_2",s_avg)
    #
    # for i in range(len(s_avg)):
    #     sheet1.cell(row=i+1, column=1).value = i+2
    #     sheet1.cell(row=i+1, column=2).value = s_avg[i]

    # func.save_excel('Clustering', woorkbook)