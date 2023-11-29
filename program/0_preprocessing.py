import os
import pandas as pd
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def contains_non_ascii(s):
    for char in s:
        if ord(char) > 127:
            return True
    return False


def generating_dataset(cert_dirs, title):
    # cert_texts including certificate's subject and issuer
    # cert_labels represent benign or malicious
    cert_texts, cert_labels = [], []

    # attack_texts including certificate's subject and issuer
    # attack_labels only malicious -> 0
    attack_texts, attack_labels = [], []

    empty_fill = 'N/A'
    for (cert_dir, num, label) in cert_dirs:
        count = 0
        for filename in os.listdir(cert_dir):
            file_path = os.path.join(cert_dir, filename)
            # read file in binary mode cause load_pem_x509_certificate takes bytes

            with open(file_path, 'rb') as cert_file:
                texts = ''
                pem_cert = cert_file.read()
                # load PEM format certificate into x509 object
                try:
                    x509_cert = x509.load_pem_x509_certificate(
                        pem_cert, default_backend())
                    count += 1
                except:
                    continue
                # extract subject and issuer principal from x509 object
                # try:
                #     texts += x509_cert.subject.get_attributes_for_oid(
                #         x509.NameOID.COMMON_NAME)[0].value
                # except:
                #     texts += sep
                # texts += ' '
                
                # issuer_attr = [x509.NameOID.COUNTRY_NAME,
                #                x509.NameOID.ORGANIZATION_NAME, x509.NameOID.COMMON_NAME]
                # for attr in issuer_attr:
                #     try:
                #         texts += x509_cert.issuer.get_attributes_for_oid(attr)[
                #             0].value
                #     except:
                #         texts += sep
                #     texts += ' '
                # if contains_non_ascii(texts) == True:
                #     print(texts, label, "\n", file_path)
                
                # main field
                subject_attr = [x509.NameOID.COMMON_NAME, x509.NameOID.COUNTRY_NAME ,x509.NameOID.ORGANIZATION_NAME ]
                for attr in subject_attr:
                    try:
                        texts += x509_cert.subject.get_attributes_for_oid(attr)[
                            0].value
                    except:
                        texts += empty_fill
                    texts += ' '
                
                # extension field
                try:
                    san_ext = x509_cert.extensions.get_extension_for_oid(x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                    if san_ext:
                        san_values = san_ext.value
                        count = 0
                        for name in san_values:
                            if isinstance(name, x509.DNSName):
                                texts += name.value
                                texts += ' '
                                count += 1
                                if (count > 2):
                                    break
                except:
                    pass
                
                if contains_non_ascii(texts):
                    continue

                if texts == ("1 1 1 1 " or '* N/A N/A * '):
                    continue

                if count > num and label == 1:
                    break
                if count > num and label == 0:
                    attack_texts.append(texts)
                    attack_labels.append(label)
                else:
                    cert_texts.append(texts)
                    cert_labels.append(label)

    df = pd.DataFrame({'text': cert_texts, 'label': cert_labels})
    df = df.drop_duplicates()
    df.to_csv(f"onlyS_{title}_train.csv", index=False)

    df2 = pd.DataFrame({'text': attack_texts, 'label': attack_labels})
    df2 = df2.drop_duplicates()
    df2.to_csv(f"onlyS_{title}_mali.csv", index=False)


if __name__ == "__main__":
    cert_dirs = [("benign-cert", 3600, 1), ("malicious-cert", 1600, 0)]
    # 3600 benign and 1600 malicious in train.csv, 536 malicious in mali.csv
    generating_dataset(cert_dirs, "NA")
