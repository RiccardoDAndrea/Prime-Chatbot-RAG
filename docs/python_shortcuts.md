# Usefull shortcuts

### Create a envoirment
---

```bash
python3 -m venv venv # <- file name
```
- Sei im Ordner drinnen wo das erstellt werden muss

**AKTIVIERUNG DES VENV**
---
```bash
source venv/bin/activate
```


### Erstellung eine `requirements.txt`
---
```bash
pip freeze > requirements.txt
```


Get you libaries from conda to a req.txt
```bash
conda list --export > requirements.txt
```

### Install req.txt 
---
```bash
pip install -r requirements.txt
```

### Merging zwei branches
---
Wenn du mergen willst musst du in **main** branch sein und dann folgenden Befehl eingeben:

```bash
git merge feature_dataset
```
Darduch nimmst du den code von `feature_dataset` und fügst in in der `main` branch

### Deleting a branch 
---

```bash
git branch -d <branch>
```

## Dockers

### Docker images erstellen
---
```
docker build -t app .
```
alternativ 

```
docker run -p 8501:8501 app
```

Docker verwendet standardmäßig den Build-Cache, um den Build-Vorgang zu beschleunigen. Wenn keine Änderungen im Dockerfile oder den darin referenzierten Dateien vorhanden sind, wird das Image aus dem Cache wieder aufgebaut, anstatt alles neu zu erstellen. Du kannst das Cache-Verhalten ändern, indem du das --no-cache Flag verwendest:
```
docker build --no-cache -t streamlit .
````

### auflistungen den Images
---
```
docker images
```

### Löschen des images
- Du kannst entweder den Namen/Tag des Images oder die Image-ID verwenden.
```
docker rmi 0895a5c6167e
```

### Auflistung der laufenden container
```
docker ps -a
```

### Container stoppen 

```
docker stop <container_id>
```

### Erzwingen des Löschens eines Images:
````
docker rmi -f <id>
