import csv


def metric_writer(path, metric, title):
    with open(path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        if isinstance(title, str):
            writer.writerow([title])
        elif isinstance(title, list):
            writer.writerow(title)
        else:
            raise ValueError("Le titre doit être une chaîne ou une liste.")

        if isinstance(metric, str):
            for line in metric.splitlines():
                writer.writerow([line])
        elif isinstance(metric, list):
            writer.writerows(metric)
        else:
            raise ValueError("Les métriques doivent être du texte ou une liste.")
