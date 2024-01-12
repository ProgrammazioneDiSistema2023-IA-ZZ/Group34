## Esecuzione da CLI
Eseguire il comando 
    cargo run --bin group_34

## Esecuzione Frontend (JS bindings)
Eseguire il comando
    npm start

## Esecuzione In Python (python bindings)

### Esecuzione su Mac
    cd backend
    if [ ! -d "./venv" ]; then
        python -m venv venv
    fi
    source myvenv/bin/activate
    maturin develop
    python

Una volta aperta la shell python ora è possibile eseguire i comandi per importare e eseguire l'ambiente

    import group_34
    group_34.main_python()

### Esecuzione su Windows
    cd backend
    if (!(Test-Path ./venv)) {
        python -m venv venv
    }
    $policy = Get-ExecutionPolicy
    Set-ExecutionPolicy Unrestricted
    ./venv\Scripts\Activate.ps1
    Set-ExecutionPolicy $policy
    maturin develop
    python

Una volta aperta la shell python ora è possibile eseguire i comandi per importare e eseguire l'ambiente

    import group_34
    group_34.main_python()