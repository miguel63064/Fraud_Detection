# 🚀 Quick Start Guide

### 1. Instalar Dependências
```bash
# Dependências principais
pip install -r requirements.txt

# Dependências de teste
pip install -r requirements-dev.txt
```

### 2. Setup de Pre-commit Hooks
```bash
bash scripts/setup-hooks.sh
```

Agora seus testes rodam automaticamente antes de cada commit! ✨

## 🧪 Rodar Testes

### Começar Rápido
```bash
# Testes rápidos (smoke)
bash scripts/run-tests.sh quick

# RESULTADO: ~30 segundos ✅
```

### Testes Completos
```bash
# Todos os testes
bash scripts/run-tests.sh all

# RESULTADO: ~3-5 minutos ✅
```

### Testes Específicos
```bash
# Apenas dados
bash scripts/run-tests.sh data

# Apenas features
bash scripts/run-tests.sh feature

# Apenas modelo
bash scripts/run-tests.sh model

# Com coverage (HTML report)
bash scripts/run-tests.sh coverage
```

### Testes em Paralelo
```bash
bash scripts/run-tests.sh parallel

# RESULTADO mais rápido: ~2 minutos ✅
```

### Debug
```bash
# Verbose com output
bash scripts/run-tests.sh debug

# Para no primeiro erro
pytest tests/ -x

# Com debugger
pytest tests/ --pdb
```

## 📊 Cobertura de Código

```bash
# Gerar relatório HTML
bash scripts/run-tests.sh coverage

# Abrir em navegador
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## 🔍 Validar antes de Commit

```bash
# Pre-commit test (rápido)
bash scripts/pre-commit-test.sh

# Se passar, pode fazer commit!
git add .
git commit -m "change"
```

## 📝 Marcar Testes

```bash
# Apenas testes de qualidade de dados
pytest tests/ -m data_quality

# Apenas testes de modelo
pytest tests/ -m model_quality

# Todos exceto slow tests
pytest tests/ -m "not slow"
```

## 🎯 GitHub Actions

O CI/CD roda automaticamente em:
- ✅ Push para `main` ou `develop`
- ✅ Pull Request
- ✅ Daily schedule (2 AM UTC)

## ✅ Checklist Antes de Fazer Commit

```bash
# 1. Run local tests
bash scripts/pre-commit-test.sh

# 2. Format código
black src/ tests/

# 3. Lint
flake8 src/ tests/

# 4. Commit
git add .
git commit -m "feat: descrição"

# 5. Push
git push origin main
```

## 📋 Arquivos Importantes

| Arquivo | Descrição |
|---------|-----------|
| `pytest.ini` | Configuração do pytest |
| `tests/conftest.py` | Fixtures compartilhadas |
| `tests/test_*.py` | Arquivos de testes |
| `.github/workflows/ci.yml` | GitHub Actions CI/CD |
| `scripts/run-tests.sh` | Script para rodar testes |
| `scripts/pre-commit-test.sh` | Script pré-commit |
| `TESTING.md` | Documentação detalhada |
| `.github/CI-CD.md` | Documentação CI/CD |

## 🆘 Troubleshooting

### Tests pass locally mas falham em CI
```bash
# Verificar Python version
python --version

# Reinstalar dependências
pip install -r requirements.txt --force-reinstall
pip install -r requirements-dev.txt --force-reinstall

# Rodar tests novamente
bash scripts/run-tests.sh all
```

### Importação falha
```bash
# Adicionar ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/fraud_project"

# Ou testar imports
python -c "import src.models; print('OK')"
```

### Pre-commit hook não roda
```bash
# Verificar permissions
ls -la .git/hooks/pre-commit

# Fazer executable
chmod +x .git/hooks/pre-commit

# Testar
bash scripts/pre-commit-test.sh
```

## 📚 Referências

- [Pytest Docs](https://docs.pytest.org/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [TESTING.md](../TESTING.md) - Documentação completa
- [CI-CD.md](./.github/CI-CD.md) - Pipeline detalhado



