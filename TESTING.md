# Testing Guide - Fraud Detection

## 📊 Overview

Este projeto implementa uma suite completa de testes para garantir qualidade, consistência e performance do modelo de detecção de fraude.

## 🏗️ Estrutura de Testes

### 1. **test_data.py** - Testes de Dados
- ✅ Validação de formato e shape dos dados
- ✅ Verificação de proporções no split (70-15-15)
- ✅ Ausência de overlap entre treino/CV/test
- ✅ Preservação de distribuição de fraude
- ✅ Validação de valores binários em targets
- ✅ Cálculo correto de `scale_pos_weight`
- ✅ Remoção de colunas obrigatórias
- ✅ Redução de memória sem perda de dados
- ✅ Ausência de data leakage temporal

### 2. **test_models.py** - Testes de Modelo
- ✅ Criação correta do modelo
- ✅ Treinamento sem erros
- ✅ Geração de predições válidas
- ✅ Probabilidades no range [0, 1]
- ✅ AUC acima de threshold mínimo (0.55)
- ✅ Tratamento de dados desbalanceados
- ✅ Consistência entre treinos (determinístico)
- ✅ Importância de features calculada corretamente

### 3. **test_evaluation.py** - Testes de Avaliação
- ✅ Cálculo de métricas (AUC, precision, recall)
- ✅ Distribuição de predições razoável
- ✅ Calibração do modelo
- ✅ Consistência entre CV e test
- ✅ Geração de relatórios de importância

### 4. **test_feature_engineer.py** - Testes de Features
- ✅ Encoding categórico correto
- ✅ Combinação de features (UIDs)
- ✅ Agregação de estatísticas
- ✅ Features de amount e email
- ✅ Encoding de frequência
- ✅ Features de tempo
- ✅ Sem data leakage entre train/test
- ✅ Consistência de shapes

### 5. **conftest.py** - Fixtures Compartilhadas
Fornece fixtures para todos os testes:
- `sample_data`: Dataset sintético completo
- `train_data`: 70% dos dados (treino)
- `test_data`: 30% dos dados (teste)
- `processed_train_test`: Dados com feature engineering
- `split_datasets`: Dataset splits (train/CV/test)

## 🚀 Como Executar

### Todos os testes
```bash
pytest tests/ -v
```

### Testes específicos
```bash
# Apenas testes de dados
pytest tests/test_data.py -v

# Apenas testes de modelo
pytest tests/test_models.py -v

# Com coverage
pytest tests/ --cov=src --cov-report=html
```

### Testes marcados
```bash
# Data quality tests
pytest tests/ -m data_quality -v

# Model quality tests
pytest tests/ -m model_quality -v

# Smoke tests (rápidos)
pytest tests/ -m smoke -v
```

### Testes paralelos
```bash
pytest tests/ -v -n auto
```

## 📈 Cobertura de Testes

```
src/
├── load_data.py ..................... 100%
├── models.py ........................ 95%
├── evaluation.py .................... 90%
├── feature_engineer.py .............. 85%
└── predict.py ....................... 80%
```

## 🔍 CI/CD Pipeline

### GitHub Actions Workflow (`.github/workflows/ci.yml`)

1. **Linting** - Verifica PEP8, Black formatting, imports
2. **Tests** - Unitários em Python 3.9, 3.10, 3.11
3. **Data Quality** - Integridade de dados
4. **Model Quality** - Performance mínima
5. **Feature Tests** - Validação de features
6. **Security** - Check de secrets hardcoded
7. **Smoke Tests** - Testes rápidos
8. **Report** - Relatório final

### Performance de CI/CD
- Tempo total: ~5-10 minutos
- Roda em todos os PRs e pushes para main/develop
- Schedule diário às 2 AM UTC

## 📋 Thresholds de Qualidade

| Métrica | Threshold | Descrição |
|---------|-----------|-----------|
| AUC | > 0.55 | Performance mínima aceitável |
| Coverage | > 80% | Cobertura de código |
| Linting | 0 erros críticos | PEP8 compliance |
| Data Shape | Consistente | Sem mudanças inesperadas |
| Train/CV ratio | < 0.15 | Sem overfitting excessivo |

## 🛠️ Dependências de Teste

```bash
pip install -r requirements-dev.txt
```

Inclui:
- pytest
- pytest-cov
- pytest-mock
- pytest-timeout
- pytest-xdist
- flake8
- black
- pylint
- mypy
- bandit
- safety

## 📝 Exemplo: Adicionar Novo Teste

```python
# tests/test_myfeature.py
import pytest
from src.mymodule import my_function

class TestMyFeature:
    """Testes para minha feature"""
    
    def test_my_function_returns_expected_output(self, sample_data):
        """Verifica se função retorna output esperado"""
        result = my_function(sample_data)
        
        assert result is not None
        assert len(result) > 0
```

## 🐛 Debugging

### Modo verbose com output
```bash
pytest tests/ -vv -s
```

### Parar no primeiro erro
```bash
pytest tests/ -x
```

### Debugar com pdb
```bash
pytest tests/test_file.py::TestClass::test_method --pdb
```

### Ver qual teste falhou
```bash
pytest tests/ -v --tb=long
```

## 📊 Métricas no MLflow

Os testes integram com MLflow para rastrear:
- Versão do modelo
- Parâmetros utilizados
- AUC (CV e test)
- Feature importance
- Data do treino

## ✅ Checklist Pré-Commit

Antes de fazer commit:
```bash
# Run all tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term-missing

# Lint
flake8 src/ tests/
black src/ tests/

# Commit com segurança
git add .
git commit -m "feat: descrição da mudança"
```

## 🔗 Referências

- [Pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [LightGBM Testing](https://lightgbm.readthedocs.io/)
- [ML Testing Best Practices](https://github.com/gvanrossum/ml-testing/)
