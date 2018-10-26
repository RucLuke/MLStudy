# -*- coding: utf-8 -*-
import qrcode
import pandas as pd
import uuid
from pathlib import Path
import os


def make_qr_code(item_id, guid):
    """
    根据url 绘制二维码
    """
    url = "http://m.wz.qichecdn.com/darkhouse/app/index?no=%s" % guid
    qr = qrcode.QRCode(version=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=4,
                       border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image()
    img = img.convert("RGBA")
    print(url)
    directory = Path("../pic")
    if not directory.exists():
        os.makedirs(directory)

    img.save("../pic/%d_%s.png" % (item_id, guid))


def save_uuid():
    # 保存guid 为excel
    file_path = r'guid.xlsx'
    directory = Path(file_path)
    if not directory.exists():
        guid_array = []
        for i in range(1, 31):
            dic = {'id': i, "guid": str(uuid.uuid4()).replace("-", "")}
            guid_array.append(dic)
        writer = pd.ExcelWriter(file_path)
        df = pd.DataFrame(guid_array)
        df.to_excel(writer, columns=['id', 'guid'], index=False, encoding="utf-8", sheet_name="Sheet")
        writer.save()


if __name__ == '__main__':
    """
    方法主入口
    """
    save_uuid()
    guid_list = pd.read_excel("guid.xlsx")
    for guid_item in guid_list.values:
        make_qr_code(guid_item[0], guid_item[1])
    print("done")
