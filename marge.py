def readlines_file(file_name):
    # 行毎のリストを返す
    with open(file_name, 'r', encoding='utf-8') as file:
        return file.readlines()


def save_file(file_name, text):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)

# 読み込んだファイルをlist型で受け取る
cal1 = readlines_file('./text/kyoto-train_ja.txt')
cal2 = readlines_file('./text/kyoto-train_en.txt')

# 改行や空白文字を削除
cal1 = list(map(lambda x: x.strip(), cal1))
cal2 = list(map(lambda x: x.strip(), cal2))

# タブ区切りで並べたリストを作成
lines = ["{0}-{1}".format(line1, line2) for line1, line2 in zip(cal1, cal2)]

save_file('marge.txt', "\n".join(lines))