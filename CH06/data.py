# 데이터 확인
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 파일을 읽어옴
data = pd.read_csv("/Users/Jinwoo/JinwooWorkspace/pytorchTencho_study/CH06/CH06.csv")

if __name__ == "__main__": 
    print(data.head())  # 데이터의 첫 5행을 출력

    # 데이터셋 전체 확인
    print(data.info())  # 데이터셋의 정보를 출력

    # 데이터의 분포 확인
    data_used = data.iloc[:, 1:4]  # 개장가, 최고가, 최저가 컬럼 선택
    data_used["Close"] = data["Close"]  # 종가 컬럼 추가

    # 히스토그램을 그리기 위해 hist() 메서드 사용
    hist = data_used.hist(bins=50, figsize=(15,10))  # bins는 히스토그램의 바구니의 수, figsize는 그래프의 크기

    # 그림을 저장하기 위한 코드
    plt.savefig("./data_histograms.png")  # 현재 디렉터리에 그림 파일을 저장

    # 저장한 그림 파일의 이름 출력
    print("Histograms of the dataset have been saved as 'data_histograms.png'")