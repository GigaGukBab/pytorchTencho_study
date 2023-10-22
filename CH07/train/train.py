


# 모델 정의
model = UNet().to(device)

# 학습률 정의
learning_rate = 0.0001

# 최적화 정의
optim = Adam(params=model.parameters(), lr=learning_rate)

# 학습 루프 정의
for epoch in range(200):
    iterator = tqdm.tqdm(train_loader)
