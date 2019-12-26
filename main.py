import cv2  #OpenCVのインポート
import webbrowser
import glob

#ファイル内の画像名を取得
files = glob.glob("./sample/*")
#画像情報を取得
img2 = cv2.imread('./pic/check_nothing.JPG')
good_sum=[]

source=[[0,"https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwjlpsOeudrkAhW-yosBHZ6ADegQjhx6BAgBEAI&url=https%3A%2F%2Fwww.amazon.co.jp%2Faventure-bleu-%25E5%2588%259D%25E5%259B%259E%25E9%2599%2590%25E5%25AE%259A%25E7%259B%25A4-%25E7%2589%25B9%25E5%2585%25B8%25E3%2581%25AA%25E3%2581%2597-DVD%2Fdp%2FB077BC69LC&psig=AOvVaw1LQGstPqnB2h263uJlntsr&ust=1568898697094150"]
        ,[0,"https://www.google.com/url?sa=i&source=imgres&cd=&cad=rja&uact=8&ved=2ahUKEwjv_daNw9rkAhVxF6YKHcZtA9MQjhx6BAgBEAI&url=https%3A%2F%2Fwww.amazon.co.jp%2F%25E9%25BC%2593%25E5%258B%2595%25E3%2582%25A8%25E3%2582%25B9%25E3%2582%25AB%25E3%2583%25AC%25E3%2583%25BC%25E3%2582%25B7%25E3%2583%25A7%25E3%2583%25B3-%25E5%2588%259D%25E5%259B%259E%25E9%2599%2590%25E5%25AE%259A%25E7%259B%25A4-CD-DVD-%25E5%2586%2585%25E7%2594%25B0%25E7%259C%259F%25E7%25A4%25BC%2Fdp%2FB07R51YQZP&psig=AOvVaw3ggYolQGO-mP5HgPmwUe-A&ust=1568901347485378"]
        ,[0,"https://www.amazon.co.jp/Magic-Hour%E3%80%90DVD%E4%BB%98%E9%99%90%E5%AE%9A%E7%9B%A4%E3%80%91-CD-DVD-PHOTOBOOK/dp/B07B12Y56T/ref=tmm_acd_title_1?_encoding=UTF8&qid=&sr="]
        ,[0,"https://www.amazon.co.jp/%E3%81%8B%E3%82%89%E3%81%A3%E3%81%BD%E3%82%AB%E3%83%97%E3%82%BB%E3%83%AB-%E5%88%9D%E5%9B%9E%E9%99%90%E5%AE%9A%E7%9B%A4-DVD%E4%BB%98-%E5%86%85%E7%94%B0%E7%9C%9F%E7%A4%BC/dp/B00RFJR6M4/ref=pd_bxgy_15_img_2/358-9255392-8714633?_encoding=UTF8&pd_rd_i=B00RFJR6M4&pd_rd_r=d8e64862-dda0-4ac5-ad86-e6a41cac9fe6&pd_rd_w=cnGCm&pd_rd_wg=Ksj6U&pf_rd_p=2d39d87c-5ff4-47a9-a2d0-79fb936a2d97&pf_rd_r=EC7G9W10ZK9NEXRF5M8K&psc=1&refRID=EC7G9W10ZK9NEXRF5M8K"]
        ,[0,"https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwioo5Wqs9rkAhUly4sBHbK9DD8QjRx6BAgBEAQ&url=https%3A%2F%2Fwww.amazon.co.jp%2Fyouthful-beautiful-%25E5%2588%259D%25E5%259B%259E%25E9%2599%2590%25E5%25AE%259A%25E7%259B%25A4-CD-DVD%2Fdp%2FB07GB3GHYN&psig=AOvVaw2sSiygcNVP2-ADJbJXRPEQ&ust=1568896929568752"]
        ,[0,"https://www.amazon.co.jp/%E5%86%85%E7%94%B0%E7%9C%9F%E7%A4%BC5th%E3%82%B7%E3%83%B3%E3%82%B0%E3%83%AB-INTERSECT-%E5%88%9D%E5%9B%9E%E9%99%90%E5%AE%9A%E7%9B%A4-DVD%E4%BB%98-%E5%86%85%E7%94%B0%E7%9C%9F%E7%A4%BC/dp/B06ZZ7BFTD/ref=pd_sim_15_6/358-9255392-8714633?_encoding=UTF8&pd_rd_i=B06ZZ7BFTD&pd_rd_r=ef021e1d-0228-4ba4-a6bb-c77ed49bd14f&pd_rd_w=lVRxm&pd_rd_wg=RUSLZ&pf_rd_p=db92e733-2248-4313-a3e1-d349f5cafca0&pf_rd_r=67M7EVS83XP6GPVQ3XEX&psc=1&refRID=67M7EVS83XP6GPVQ3XEX"]
        ]
    # imgと一番特徴点が多い画像探し
for i in range(len(files)):
    # img1の読み出し

    img1 = cv2.imread(files[i])
    # img2の読み出し


    # img1をグレースケールで読み出し
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # img2をグレースケールで読み出し
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # AKAZE検出器の生成
    akaze = cv2.AKAZE_create()
    # gray1にAKAZEを適用、特徴点を検出
    kp1, des1 = akaze.detectAndCompute(gray1,None)
    # gray2にAKAZEを適用、特徴点を検出
    kp2, des2 = akaze.detectAndCompute(gray2,None)

    # BFMatcherオブジェクトの生成
    bf = cv2.BFMatcher()

    # Match descriptorsを生成
    matches = bf.knnMatch(des1, des2, k=2)

    #より似ている特徴点の抽出する配列
    good = []
    #倍率をかけてより似ている点を抽出する
    for m,n in matches:
        if m.distance <0.75*n.distance:
            good.append([m])
    good_sum.append(len(good))
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('match'+str(i)+'.jpg',img3)
for i in range(len(good_sum)):

    source[i][0]=good_sum[i]

# matchesをdescriptorsのdistance順(似ている順)にsortする
#match = sorted(good, key = lambda x:x.distance)


source.sort(key=lambda x:x[0],reverse=True)

#img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#cv2.imwrite('match.jpg',img3)


if source[0][0]<5:
    print("参照した画像の中に商品は存在しません")
    exit(0)
#ChromeDriverのパスを引数に指定しChromeを起動
url =source[0][1]
webbrowser.open(url)
