import matplotlib.pyplot as plt

labels = ['Healthy', 'Heart Disease']
matrix = [[57, 24],[17, 82]]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(matrix, cmap='Reds')
plt.title('Random Forest Confusion Matrix')
fig.colorbar(cax, boundaries=range(5,100))
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
