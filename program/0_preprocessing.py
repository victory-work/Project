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
    cert_texts = []
    # cert_labels represent benign or malicious
    cert_labels = []
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
                    if count == num:
                        break
                except:
                    continue
                # extract subject and issuer principal from x509 object
                try:
                    texts += x509_cert.subject.get_attributes_for_oid(
                        x509.NameOID.COMMON_NAME)[0].value
                except:
                    texts += 'N/A'
                texts += ' '
                issuer_attr = [x509.NameOID.COUNTRY_NAME,
                               x509.NameOID.ORGANIZATION_NAME, x509.NameOID.COMMON_NAME]
                for attr in issuer_attr:
                    try:
                        texts += x509_cert.issuer.get_attributes_for_oid(attr)[
                            0].value
                    except:
                        texts += 'N/A'
                    texts += ' '
                if contains_non_ascii(texts) == True:
                    print(texts, label, "\n", file_path)
                cert_texts.append(texts)
                cert_labels.append(label)
    df = pd.DataFrame({'text': cert_texts, 'label': cert_labels})
    df.to_csv(title, index=False)


if __name__ == "__main__":
    cert_dirs = [("benign-cert", 9999, 1), ("malicious-cert", 2000, 0)]
    # cert_dirs = [("adversasry_malicious", 200, 0)]
    generating_dataset(cert_dirs, "cert_dataset.csv")
