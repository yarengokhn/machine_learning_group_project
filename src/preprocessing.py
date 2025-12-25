import re


def preprocess_code(text,is_Code=True):
    if text is None: return ""
  
    if is_Code:
        # Yorumları kaldır
        text = re.sub(r'#.*', '', text)
        # Fazla boşlukları kaldır
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('\n', '').strip()
  # 3. CamelCase ve snake_case parçalama (Tokenization öncesi anlamı güçlendirir)
    if is_Code:
        # Örn: calculateAverageScore -> calculate Average Score
        text = re.sub('([a-z0-9])([A-Z])', r'\1 \2', text)
        # Örn: calculate_average -> calculate average
        text = text.replace('_', ' ')
    return text.lower() # Modelin öğrenmesini kolaylaştırmak için küçük harf