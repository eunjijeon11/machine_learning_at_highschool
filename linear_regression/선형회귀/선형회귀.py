from matplotlib.pylab import plot, show, scatter

def main():
    X = [1,2,3,4,5,6,7,8] #X값 입력
    Y = [3,6,5,9,10,14,18,20] #Y값 입력
    N = 8 #자료 수
    
    w, b = do_linear_regression(X, Y, N)
    print(w,b) #기울기, 절편 출력
    y = [(w * number + b) for number in X]
    #기울기, 절편을 기반으로 작성한 일차함수
    scatter(X,Y)
    plot(X,y)
    show() #시각화

def do_linear_regression(X, Y, N, rate=0.001, epochs=10000):
    w_0 = 1.0 #초기 기울기
    b_0 = 1.0 #초기 y절편
    for t in range(epochs):
        #경사하강법
        w_grad, b_grad = gradient_descendant(X,Y,N,w_0,b_0)
        w_0 -= rate * w_grad 
        b_0 -= rate * b_grad
    return w_0, b_0 #최종 기울기, y절편

def gradient_descendant(X,Y,N,w_0,b_0):
    w_grad = 0
    b_grad = 0
    for i in range(N):
        w_grad += X[i] * (X[i] * w_0 + b_0 -Y[i])
        b_grad += (X[i] * w_0 + b_0 - Y[i])
    return w_grad, b_grad #편미분 결과
    

if __name__ == "__main__":
    main()
