# Dinâmicas de Participação de Debates Parlamentares em Portugal

## Descrição / Resumo
Este projeto propõe uma abordagem computacional para analisar a estrutura das interações nos debates parlamentares portugueses. Ao invés de focar apenas no conteúdo semântico, o objetivo principal é analisar a dinâmica sequencial: quem fala com quem, em que ordem e com que frequência. Utilizando o corpus PTPARL-D (1976–2019), o sistema mapeia a dimensão estrutural e relacional do plenário recorrendo a **Modelos de Linguagem de Grande Escala (LLMs)** e modelação analítica baseada na teoria dos *participation shifts* de Gibson.

## Funcionalidades Principais
* **Recolha e Transformação de Dados:** Conversão em larga escala de ficheiros do corpus agrupados em `XML` para estruturas sequenciais e manipuláveis em `JSON`.
* **Pré-processamento e Limpeza Textual:** Normalização de identificadores (oradores, papéis institucionais, partidos), tratamento de ruído e segmentação da conversa em unidades analíticas autónomas (*utterances*).
* **Anotação Estruturada por LLM:** Inferência avançada de destinatários (*targets*) de falas, mesmo quando não mencionados de forma explícita, recorrendo a tarefas de Engenharia de Prompt.
* **Análise Sequencial de Interações:** Automatização computacional para classificar e visualizar transcrições segundo os 13 tipos de *participation shifts* de Gibson.
* **Extração de Métricas Conversacionais:** Cálculo quantitativo de taxas de alternância de turno, continuidade discursiva, dominância discursiva (individual e partidária) e propensão de resposta ou interrupção.
* **Avaliação de Concordância (Agreement):** Implementação de medições robustas (*Krippendorff’s Alpha*, *Cohen’s Kappa*) para avaliar a concordância entre múltiplos anotadores (Humanos e Inteligência Artificial).

## Tecnologias Utilizadas
* **Linguagens e Estruturações:** Python, JSON, XML
* **Bibliotecas / Frameworks NLP:** ParShift (para implementar o modelo quantitativo analítico de *participation shifts*)
* **Inteligência Artificial & LLMs:** ChatGPT, Microsoft Copilot, Gemini, Claude

## Arquitetura e Metodologia
O projeto desenrola-se em torno de um *pipeline* metodológico estruturado nas seguintes etapas:
1. **Recolha e estruturação:** Identificação dos debates do corpus e abstração por metadados de legislaturas e datas.
2. **Transformação e modelação:** Conversão da estrutura discursiva de `XML` para atributos chave contidos num objecto em `JSON`.
3. **Limpeza e normalização:** Distinção entre categorias estritas (ex: oradores formais vs discurso institucional).
4. **Anotação Tripartida:** Incorporação da variável "destinatário" no encadeamento de ações, testada de iterativamente baseada na concordância humana e de um LLM-Avaliador.
5. **Avaliação de concordância:** Refinamento lógico e consistência garantidos através de métricas de fiabilidade com base na classificação de pares avaliadores.
6. **Métricas e modulação iteracional:** Estatística descritiva traduzida em matrizes de interação e grafos de distribuição temporal por bancadas ou intervenientes.

## Instalação e Uso
```bash
git clone https://github.com/guilherme-puga-a22306246/AISIC-Lab-Portuguese-Parliament

# Criar o ambiente
python -m venv venv

# Ativar o ambiente (Linux/macOS)
source venv/bin/activate

# Ativar o ambiente (Windows)
venv\Scripts\activate

#instalar dependências
pip install -r requirements.txt
```

## Autores / Equipe
* **Equipa de Investigação/Desenvolvimento:** Guilherme Puga e Miguel Gouveia (LEI)
* **Orientação:** Manuel Pita (Orientador) e Bruno Saraiva (Coorientador)
* **Parcerias Analíticas e Institucionais:** AISIC Lab – Artificial Intelligence, Social Interaction and Complexity, CICANT da Universidade Lusófona
* **Instituição de Ensino:** Universidade Lusófona, Centro Universitário de Lisboa - Departamento de Engenharia Informática e Sistemas de Informação (DEISI)
