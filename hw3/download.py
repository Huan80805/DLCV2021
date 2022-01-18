import gdown

id = ['1HGyTouIuFZQI_kcKDzjIZeqHpQiDwKpf','13YBAoPODWRN8QDBzAftSUBxt9DodcXKT','1K8BcYrHuwcLQMWu1bS5OTFJhShQmWgFK']
output = ['best.pth', 'best2.pth', 'best6.pth']
for i in range(len(id)):
    gdown.download(id=id[i], output=output[i], quiet=False)