
"""MPI híbrido: distribuir tableros y ramas entre grupos de procesos."""

import argparse
from pathlib import Path

from mpi4py import MPI

ROOT = Path(__file__).resolve().parent.parent


def leer_seeds(ruta):
    seeds = {}

    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()

        if not linea or linea.startswith("#"):
            continue

        identificador, contenido = linea.split("|", 1)
        valores = [
            int(c) for c in contenido.strip().replace(".", "0")
        ]

        if len(valores) != 81:
            raise ValueError(
                f"El tablero {identificador} no tiene 81 caracteres."
            )

        seeds[identificador.strip()] = [
            valores[i:i + 9] for i in range(0, 81, 9)
        ]

    return seeds


def puede_colocar(tablero, fila, columna, valor):
    if any(tablero[fila][c] == valor for c in range(9)):
        return False

    if any(tablero[f][columna] == valor for f in range(9)):
        return False

    inicio_fila = (fila // 3) * 3
    inicio_columna = (columna // 3) * 3

    for f in range(inicio_fila, inicio_fila + 3):
        for c in range(inicio_columna, inicio_columna + 3):
            if tablero[f][c] == valor:
                return False

    return True


def resolver(tablero):
    for fila in range(9):
        for columna in range(9):
            if tablero[fila][columna] != 0:
                continue

            for valor in range(1, 10):
                if puede_colocar(tablero, fila, columna, valor):
                    tablero[fila][columna] = valor

                    if resolver(tablero):
                        return True

                    tablero[fila][columna] = 0

            return False

    return True


def generar_ramas(tablero):
    for fila in range(9):
        for columna in range(9):
            if tablero[fila][columna] == 0:
                ramas = []

                for valor in range(1, 10):
                    if puede_colocar(tablero, fila, columna, valor):
                        copia = [f[:] for f in tablero]
                        copia[fila][columna] = valor
                        ramas.append(copia)

                return ramas

    return []


def resolver_rama(rama):
    if rama is None:
        return None

    copia = [fila[:] for fila in rama]

    if resolver(copia):
        return copia

    return None


def resolver_tablero(grupo, tablero):
    rank = grupo.Get_rank()
    size = grupo.Get_size()
    solucion = None

    if rank == 0:
        if all(valor != 0 for fila in tablero for valor in fila):
            copia = [fila[:] for fila in tablero]
            solucion = copia if resolver(copia) else None
            asignaciones = [[] for _ in range(size)]
        else:
            ramas = generar_ramas(tablero)
            asignaciones = [[] for _ in range(size)]

            # Distribuir todas las ramas entre los procesos.
            for indice, rama in enumerate(ramas):
                asignaciones[indice % size].append(rama)
    else:
        asignaciones = None

    ramas_locales = grupo.scatter(asignaciones, root=0)

    solucion_local = None

    for rama in ramas_locales:
        resultado = resolver_rama(rama)

        if resultado is not None and solucion_local is None:
            solucion_local = resultado

    soluciones = grupo.gather(solucion_local, root=0)

    if rank == 0:
        if solucion is not None:
            return solucion

        validas = [
            resultado for resultado in soluciones
            if resultado is not None
        ]

        return validas[0] if validas else None

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Resolver Sudokus con estrategia MPI híbrida."
    )

    parser.add_argument(
        "--archivo",
        type=Path,
        default=ROOT / "seeds.txt"
    )

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2 or size % 2 != 0:
        if rank == 0:
            print(
                "Error: se necesita un número par de procesos MPI, mínimo 2."
            )
        return

    # Crear dos grupos de procesos.
    grupo = comm.Split(color=rank % 2, key=rank)
    grupo_id = rank % 2
    rank_grupo = grupo.Get_rank()

    if rank == 0:
        try:
            ruta = args.archivo

            if not ruta.is_absolute():
                ruta = ROOT / ruta

            seeds = leer_seeds(ruta)

            if not seeds:
                raise ValueError("No se encontraron tableros.")

            error = None

        except (OSError, ValueError) as exc:
            seeds = None
            error = str(exc)
    else:
        seeds = None
        error = None

    error = comm.bcast(error, root=0)

    if error:
        if rank == 0:
            print(f"Error: {error}")
        grupo.Free()
        return

    seeds = comm.bcast(seeds, root=0)

    identificadores = sorted(
        seeds.keys(),
        key=lambda x: int(x)
    )

    # Cada grupo recibe una parte de los tableros.
    tableros_grupo = [
        identificador
        for i, identificador in enumerate(identificadores)
        if i % 2 == grupo_id
    ]

    comm.Barrier()
    inicio = MPI.Wtime()

    resultados_grupo = []

    # Dentro de cada grupo se reparten las ramas de cada tablero.
    for identificador in tableros_grupo:
        tablero = seeds[identificador]
        solucion = resolver_tablero(grupo, tablero)

        if rank_grupo == 0:
            resuelto = solucion is not None
            resultados_grupo.append((identificador, resuelto))

    # Solo el coordinador de cada grupo entrega sus resultados.
    datos = resultados_grupo if rank_grupo == 0 else None

    # El coordinador global recibe los resultados de ambos grupos.
    todos_resultados = comm.gather(datos, root=0)

    # El tiempo termina después de recibir todos los resultados.
    if rank == 0:
        tiempo_total = MPI.Wtime() - inicio

        resultados = []

        for grupo_resultados in todos_resultados:
            if grupo_resultados is not None:
                resultados.extend(grupo_resultados)

        resultados.sort(key=lambda x: int(x[0]))

        for identificador, resuelto in resultados:
            estado = "Resuelto" if resuelto else "Sin solución"
            print(f"ID: {identificador} - {estado}")

        print(f"\nTableros procesados: {len(resultados)}")
        print(f"Procesos MPI: {size}")
        print("Grupos de trabajo: 2")
        print(f"Procesos por grupo: {size // 2}")
        print(f"Tiempo total MPI: {tiempo_total:.6f} segundos")

    grupo.Free()


if __name__ == "__main__":
    main()
