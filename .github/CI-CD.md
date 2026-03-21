# CI/CD Pipeline Documentation

## 📌 Overview

Este projeto implementa um pipeline automático de CI/CD usando **GitHub Actions** que valida qualidade de código, executa testes, e garante que novos commits atendem aos padrões de qualidade.

## 🔄 Pipeline Stages

### 1️⃣ **Lint** (Linting & Format Check)
- **Duração**: ~1 min
- **O que faz**:
  - ✅ Flake8: Verifica PEP8 compliance
  - ✅ Black: Verifica formatação de código
  - ✅ isort: Verifica organização de imports
- **Falha se**: Erros críticos em PEP8 (F63, F7, F82)

### 2️⃣ **Test** (Run Tests)
- **Duração**: ~3-5 mins
- **O que faz**:
  - ✅ Executa pytest em Python 3.9, 3.10, 3.11
  - ✅ Gera coverage report (XML, HTML, terminal)
  - ✅ Upload para Codecov
- **Falha se**: Any test fails
- **Artifacts**: 
  - `coverage.xml`
  - `htmlcov/` (HTML report)

### 3️⃣ **Data Quality** (Data Quality Tests)
- **Duração**: ~1 min
- **O que faz**:
  - ✅ Valida integridade dos dados
  - ✅ Verifica shapes e tipos
  - ✅ Testa splits (70-15-15)
  - ✅ Detecta data leakage
- **Depende de**: Lint

### 4️⃣ **Model Quality** (Model Quality Tests)
- **Duração**: ~2-3 mins
- **O que faz**:
  - ✅ Treina modelo LightGBM
  - ✅ Valida AUC (>0.55)
  - ✅ Testa predições
  - ✅ Verifica calibração
- **Timeout**: 5 minutos
- **Depende de**: Lint, Test

### 5️⃣ **Feature Tests** (Feature Engineering Tests)
- **Duração**: ~1-2 mins
- **O que faz**:
  - ✅ Valida encoding categórico
  - ✅ Testa combinação de features
  - ✅ Verifica agregações
  - ✅ Detecta data leakage
- **Depende de**: Lint

### 6️⃣ **Security** (Security & Secrets Check)
- **Duração**: ~1 min
- **O que faz**:
  - ✅ Bandit: Detecta vulnerabilidades
  - ✅ Grep: Procura por secrets hardcoded
  - ✅ Safety: Verifica dependências vulneráveis
- **Artifacts**: `bandit-report.json`
- **Não falha CI**: Continue mesmo com warnings

### 7️⃣ **Smoke Tests** (Smoke Tests)
- **Duração**: ~30 segs
- **O que faz**:
  - ✅ Testes rápidos de sanidade
  - ✅ Valida imports e APIs básicas
- **Depende de**: Test

### 8️⃣ **Test Report** (Test Report)
- **O que faz**:
  - ✅ Cria summary do pipeline
  - ✅ Falha CI se jobs críticos falharem
- **Jobs críticos**:
  - Linting
  - Tests
  - Model Quality

## 📊 Triggers

O pipeline roda automaticamente em:

```yaml
# Triggers
- Push em qualquer branch (main, develop, ou feature branches)
- Pull Request contra main ou develop
- Schedule diário (2 AM UTC)
```

### Push vs PR

- **Push direto**: Todos os 8 jobs rodam
- **Pull Request**: Todos os 8 jobs rodam, porém PR não pode ser merged sem sucesso

## 🎯 Thresholds e Critérios

| Métrica | Threshold | Job |
|---------|-----------|-----|
| AUC | > 0.55 | Model Quality |
| Coverage | Reportado | Test |
| Linting | 0 erros F-series | Lint |
| Test Pass Rate | 100% | Test |
| Data Integrity | ✓ | Data Quality |
| No Secrets | ✓ | Security |

## 📝 Status Badges

Adicione ao README.md:

```markdown
[![Tests](https://github.com/seu-usuario/fraud_project/actions/workflows/ci.yml/badge.svg](https://github.com/seu-usuario/fraud_project/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/seu-usuario/fraud_project/branch/main/graph/badge.svg)](https://codecov.io/gh/seu-usuario/fraud_project)
```

## 🚀 Local Development

### Instalar dependências
```bash
pip install -r requirements-dev.txt
```

### Rodar testes localmente
```bash
# Todos os testes
pytest tests/ -v

# Com coverage
pytest tests/ --cov=src --cov-report=html

# Testes paralelos
pytest tests/ -v -n auto
```

### Linting local
```bash
# Check style
black --check src/ tests/

# Auto-format
black src/ tests/

# Linting
flake8 src/ tests/ --max-line-length=120

# isort
isort src/ tests/
```

### Pre-commit hooks
```bash
# Setup
bash scripts/setup-hooks.sh

# Agora todos os commits rodarão testes automaticamente!
```

## 📋 Debugging CI Failures

### 1. Verificar logs
```
GitHub Actions → seu-workflow → seus-job → Build logs
```

### 2. Reproduzir localmente
```bash
# Mesmo Python version
python -V

# Instalar exatas mesmas dependências
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Rodar mesmo comando do workflow
pytest tests/ --cov=src --cov-report=xml
```

### 3. Testes individuais
```bash
# Um arquivo
pytest tests/test_models.py -v

# Uma classe
pytest tests/test_models.py::TestModelTraining -v

# Um teste específico
pytest tests/test_models.py::TestModelTraining::test_model_trains_successfully -v
```

## 🔐 Secrets Management

O workflow não usa secrets (não são necessários para repo público).

Se precisar adicionar credenciais:
1. GitHub → Settings → Secrets and variables → Actions
2. Criar novo secret
3. Usar em workflow: `${{ secrets.SECRET_NAME }}`

## 📈 Monitoring

### Success
- ✅ Todos os jobs em verde
- ✅ PR pode fazer merge
- ✅ Badge mostra sucesso

### Failure
- ❌ Job falha
- ❌ PR bloqueia merge
- ❌ Notificação automática (se configurado)

### Rerun
Se falha transitória (network, etc):
```
GitHub Actions → seu-workflow → Re-run failed jobs
```

## 🎯 Best Practices

1. **Commit frequente**: Pequenos commits são testados rapidamente
2. **PR reviews**: Aguarde CI passar antes de review
3. **Branch protection**: Configure para exigir CI verde
4. **Monitor coverage**: Mantenha > 80%
5. **Fix fast**: Quando CI falha, fix é prioritário

## 📞 Troubleshooting

### "Tests pass locally but fail in CI"
- ✅ Diferentes versões de Python?
- ✅ Diferentes dependências de sistema?
- ✅ Dados diferentes nos runners?
- ✅ Paths absolutas vs relativas?

### "Timeout em testes"
- ✅ Aumentar timeout em `pytest.ini`
- ✅ Otimizar testes lentos
- ✅ Usar `-n auto` para paralelizar

### "Coverage drops"
- ✅ Adicionar testes para novo código
- ✅ Evitar dead code
- ✅ Monitorar trends no Codecov

## 📚 Referências

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov](https://codecov.io/)
- [ML Testing Best Practices](https://evals.phd/evals-best-practices)
