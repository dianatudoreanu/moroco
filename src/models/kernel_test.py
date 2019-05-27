
from src.utils.paths_utils import get_absolute_path
from src.dataset.dataset_utils import DatasetUtils
import numpy as np

lens = 50000

def get_sk():
    sk_matrix = np.load(get_absolute_path('@data/K6.npy')).astype(np.float32)

    dp = np.reshape(np.diag(sk_matrix), (sk_matrix.shape[0], 1))
    dp_t = dp.T
    
    num = np.matmul(dp, dp_t) 
    num[num == 0] = 1

    sk_matrix = sk_matrix/np.sqrt(num)

    print(num.shape, sk_matrix.shape)

    return sk_matrix

sk_matrix = get_sk()

train_data, train_labels, train_sk_indices = DatasetUtils.read_dataset(get_absolute_path("@data/subtask1/train_dataset.txt") , return_tuples=False)
val_data, val_labels, val_sk_indices = DatasetUtils.read_dataset(get_absolute_path("@data/subtask1/val_dataset.txt") , return_tuples=False)

train_labels = np.array([[1, -1] if x == 'MD' else [-1, 1] for x in train_labels[:lens]])
val_labels = np.array([[1, -1] if x == 'MD' else [-1, 1] for x in val_labels])

lam = 0.00001
train_sk_indices = train_sk_indices[:lens]

print(len(train_sk_indices), len(val_sk_indices))
n = len(train_sk_indices)
train_krr_data = np.array(list(map( lambda x : sk_matrix[x][train_sk_indices], train_sk_indices))).astype(np.float32)
val_krr_data = np.array(list(map( lambda x : sk_matrix[x][train_sk_indices], val_sk_indices))).astype(np.float32)

print('before training')
(w,a,b,c) =  np.linalg.lstsq(train_krr_data + np.eye(n) * n * lam, train_labels)

preds = np.argmax(np.matmul(train_krr_data, w), 1)

ans = np.argmax(train_labels, 1)
acc = np.sum(preds == ans) / ans.shape[0]


val_preds = np.argmax(np.matmul(val_krr_data, w), 1)
val_ans = np.argmax(val_labels, 1)
val_acc = np.sum(val_preds == val_ans) / val_ans.shape[0]


print(f'Train Accuracy is {acc}')
print(f'Val Accuracy is {val_acc}')

# print('before training')
# # model = KernelRidge(alpha = 0.0001)
# # model.fit(train_krr_data, train_labels)

# model  = SVC()
# model.fit(train_krr_data, train_labels)

# print('after training')
# preds = model.predict(val_krr_data)

# preds[preds <= 0] = -1
# preds[preds > 0] = 1

# print(preds)
# accuracy = np.sum(preds == val_labels)/val_labels.shape[0]
# print(accuracy)






