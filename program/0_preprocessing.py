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

def get_cert_text(file_path):
    # print("*"*60)
    empty_fill = '""'
    with open(file_path, 'rb') as cert_file:
        x509_cert = ""
        cert_content = cert_file.read()
        # load PEM format certificate into x509 object
        try:
            file_extension = file_path.split('.')[-1]
            if file_extension == "pem":
                x509_cert = x509.load_pem_x509_certificate(cert_content, default_backend())   
            elif file_extension == "crt":
                x509_cert = x509.load_der_x509_certificate(cert_content, default_backend()) 
            
            else:
                print("file_extension no found:",file_extension)
        except:
            print("error: can't load cert")
            print(file_path)
            return ""
        
        # main field
        texts = ''
        subject_attr = [x509.NameOID.COMMON_NAME, x509.NameOID.COUNTRY_NAME ,x509.NameOID.ORGANIZATION_NAME ]
        for attr in subject_attr:
            try:
                texts += x509_cert.subject.get_attributes_for_oid(attr)[
                    0].value
            except:
                texts += empty_fill
            texts += ' '
        
        # extension field
        # try:
        #     san_ext = x509_cert.extensions.get_extension_for_oid(x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        #     if san_ext:
        #         san_values = san_ext.value
        #         count = 0
        #         for name in san_values:
        #             if isinstance(name, x509.DNSName):
        #                 texts += name.value
        #                 texts += ' '
        #                 count += 1
        #                 if (count > 2):
        #                     break
        # except:
        #     texts += empty_fill
        
        if contains_non_ascii(texts) or (texts == '"" "" "" "" ') or ("1 1 1 1 " in texts):
            return ""
        return texts


def generating_dataset(cert_dir):
    # cert_texts including certificate's subject and issuer
    # cert_labels represent benign or malicious
    cert_texts, cert_labels = [], []
    
    count = 0 # count the
    dir_name, num, label = cert_dir 
    
    if dir_name == "benign-cert": # label = 1
        print("benign-cert","*"*60)
        for filename in os.listdir(dir_name):
            file_path = os.path.join(dir_name, filename)
            cert_text = get_cert_text(file_path)
            
            if cert_text != "" and count < num:
                count+=1
                cert_texts.append(cert_text)
                cert_labels.append(label)
            
            if count >= num:
                break
        
        benign_df = pd.DataFrame({'text': cert_texts, 'label': cert_labels})
        benign_df = benign_df.drop_duplicates()
        print("cert get:",benign_df.shape[0])
        return benign_df
    
    if dir_name == "malicious-cert": # label = 0
        print("malicious-cert","*"*60)
        # attack_texts including certificate's subject and issuer
        # attack_labels only malicious -> 0
        attack_texts, attack_labels = [], []
        
        for filename in os.listdir(dir_name):
            file_path = os.path.join(dir_name, filename)
            cert_text = get_cert_text(file_path)
            
            if cert_text != "" and count < num:
                count+=1
                cert_texts.append(cert_text)
                cert_labels.append(label)
                
            if count >= num:
                attack_texts.append(cert_text)
                attack_labels.append(label)
        
        attack_df = pd.DataFrame({'text': attack_texts, 'label': attack_labels})
        attack_df = attack_df.drop_duplicates()
        print("target set:", attack_df.shape[0])
        attack_df.to_csv(f"double_quotes_mali.csv", index=False)
        
        mali_df = pd.DataFrame({'text': cert_texts, 'label': cert_labels})
        mali_df = mali_df.drop_duplicates()
        print("cert get:",mali_df.shape[0])
        return mali_df
    
    if dir_name ==  "new_mali_cert":
        print("new_malicious_cert","*"*60)
        for filename in os.listdir(dir_name):
            file_path = os.path.join(dir_name, filename)
            cert_text = get_cert_text(file_path)
            
            if cert_text != "" and count < num:
                count+=1
                cert_texts.append(cert_text)
                cert_labels.append(label)
                continue
            
            if count >= num:
                break
        
        new_mali_df = pd.DataFrame({'text': cert_texts, 'label': cert_labels})
        new_mali_df = new_mali_df.drop_duplicates()
        print("cert get:",new_mali_df.shape[0])
        return new_mali_df
    
if __name__ == "__main__":
    # 3600 benign and 1300 malicious in train.csv, 536 malicious in mali.csv
    cert_dirs = [("benign-cert", 3600, 1), ("malicious-cert", 1100, 0), ("new_mali_cert",3708,0)]
    
    benign_df = generating_dataset(cert_dirs[0])   # all cert for train
    mali_df = generating_dataset(cert_dirs[1])     # some cert for attack
    new_mali_df = generating_dataset(cert_dirs[2]) # all cert are for train
    # benign_df.to_csv("benign.csv",index=False)
    # mali_df.to_csv("mali.csv",index=False)
    # new_mali_df.to_csv("new_mali.csv",index=False)
    
    print("*"*60)
    train_df = pd.concat([benign_df, mali_df,new_mali_df],axis=0)
    train_df = train_df.drop_duplicates()
    print("total train cert get:",train_df.shape[0])
    train_df.to_csv("double_quotes_train.csv",index=False)
    
    
