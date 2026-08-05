import smtplib
from email.mime.text import MIMEText

import cv2


def mailToMeWithImage(zhuti, mail_txt, image_np, sender, receiver, password):
    """
    发送带图片附件的告警邮件
    :param zhuti:     邮件标题
    :param mail_txt:  邮件正文（文字）
    :param image_np:  要发送的图片，OpenCV格式，即 numpy.ndarray (BGR)
    :param sender:    发件人163邮箱地址
    :param receiver:  收件人邮箱地址
    :param password:  163邮箱客户端授权码（非登录密码）.
    """
    import io
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart

    # 创建一个带附件的邮件对象（MIMEMultipart）
    msg = MIMEMultipart()
    msg["Subject"] = zhuti
    msg["From"] = sender
    msg["To"] = receiver

    # 邮件正文（纯文本）
    text_part = MIMEText(mail_txt, "plain", "utf-8")
    msg.attach(text_part)

    try:
        # 将 OpenCV 图像（BGR）转换为 RGB，再转 JPEG 格式的二进制数据
        from PIL import Image

        # OpenCV图像是 BGR，先转成 RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 保存到内存中的字节流（不需要存硬盘）
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="JPEG")  # 也可以选 PNG
        img_byte_arr.seek(0)

        # 构造图片附件
        image = MIMEImage(img_byte_arr.read())
        image.add_header("Content-Disposition", "attachment", filename="detection_alert.jpg")
        msg.attach(image)

        # 连接SMTP服务器并发送
        with smtplib.SMTP_SSL("smtp.163.com", 465, timeout=10) as smtp:
            smtp.login(sender, password)
            smtp.sendmail(sender, [receiver], msg.as_string())
            print("✅ 带图片的邮件发送成功！")
    except Exception as e:
        print(f"❌ 带图片的邮件发送失败，错误详情: {e}")
        raise
