from cryptography import x509
from cryptography.hazmat.backends import default_backend

# Assuming you have a PEM or DER encoded X.509 certificate
# You can load the certificate using cryptography library
with open("malicious-cert\d54b3bff7196c2201a3fe60fac8aa7601175db092268227b8b72c33910fc9dc5.pem", "rb") as cert_file:
    cert_data = cert_file.read()
    cert = x509.load_pem_x509_certificate(cert_data, default_backend())
    #cert = x509.load_der_x509_certificate(cert_data, default_backend())
# Extracting subject information
subject = cert.subject

# Extracting various fields using NameOID constants
common_name = subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
organization = subject.get_attributes_for_oid(x509.NameOID.ORGANIZATION_NAME)
organizational_unit = subject.get_attributes_for_oid(x509.NameOID.ORGANIZATIONAL_UNIT_NAME)
country = subject.get_attributes_for_oid(x509.NameOID.COUNTRY_NAME)
state = subject.get_attributes_for_oid(x509.NameOID.STATE_OR_PROVINCE_NAME)
locality = subject.get_attributes_for_oid(x509.NameOID.LOCALITY_NAME)
email = subject.get_attributes_for_oid(x509.NameOID.EMAIL_ADDRESS)
entetion = subject.get_attributes_for_oid(x509.Extension)

# Print the extracted information
print(f"Common Name: {common_name[0].value}")
print(f"Organization: {organization[0].value}")

if organizational_unit:
    print(f"Organizational Unit: {organizational_unit[0].value}")
if country:
    print(f"Country: {country[0].value}")

if state:
    print(f"State: {state[0].value}")

if locality:
    print(f"Locality: {locality[0].value}")

if email:
    print(f"Email: {email[0].value}")

# Extract and print Subject Alternative Names (DNS names)
san_ext = cert.extensions.get_extension_for_oid(x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
san_values = san_ext.value

# Check if the extension is a x509.DNSName type and print the values
for name in san_values:
    if isinstance(name, x509.DNSName):
        print(f"DNS Name: {name.value}")

print('"" "" "" "" ')