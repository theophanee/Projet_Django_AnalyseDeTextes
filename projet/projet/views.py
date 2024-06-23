from django.http import HttpResponse
from django.shortcuts import render
from .text_processing import TextProcessor  # Importez la classe TextProcessor
from django.http import JsonResponse


from django.core.cache import cache

class View:
    ngrames = None

    def home_view(request):
        context = {
            'name': 'John',
            'age': 30,
        }
        return render(request, 'home.html', context)


    def appliquer_options(request):
        text_processor = TextProcessor()

        if request.method == 'POST':
            # Récupérer les données envoyées depuis le front-end
            largeur = request.POST.get('largeur')
            alignement = request.POST.get('alignement')
            fichiers = request.FILES.getlist('fichiers')
            urls = request.POST.getlist('urls')
            programme = request.POST.get('programme')

            resultats_justification = []

            # Fonction pour justifier le texte en fonction du programme choisi
            def justify_text(text, programme):
                if programme == 'glouton':
                    return text_processor.justify_text_greedy(text, int(largeur), alignement)
                else:
                    return text_processor.justify_text_dynamic(text, int(largeur), alignement)

            # Traitement des fichiers
            for fichier in fichiers:
                # Générer une clé unique pour chaque fichier en utilisant son nom, alignement, largeur et programme
                cache_key = f"text_{fichier.name}_{alignement}_{largeur}_{programme}"
                contenu_fichier = fichier.read().decode("utf-8")
                # Vérifier si le résultat de l'opération de justification est déjà mis en cache
                resultats_cache_key = f"resultats_{cache_key}"
                resultats_justification_cache = cache.get(resultats_cache_key)
                if not resultats_justification_cache:
                    # Si les résultats ne sont pas en cache, justifier le texte
                    texte_cache_key = f"text_{cache_key}"
                    texte_cache = cache.get(texte_cache_key)
                    if not texte_cache:
                        texte_cache = contenu_fichier
                        # Mettre en cache le texte pour une durée déterminée
                        cache.set(texte_cache_key, texte_cache, timeout=3600)
                    else:
                        contenu_fichier = texte_cache
                    resultats = justify_text(contenu_fichier, programme)
                    # Mettre en cache les résultats de justification
                    cache.set(resultats_cache_key, resultats, timeout=3600)
                else:
                    resultats = resultats_justification_cache
                resultats_justification.append(resultats)

            # Traitement des URLs
            for url in urls:
                if url.strip():  # Vérifie si l'URL n'est pas vide après suppression des espaces
                    # Générer une clé unique pour chaque URL en utilisant son lien, alignement et largeur
                    cache_key = f"text_{url}_{alignement}_{largeur}_{programme}"
                    # Vérifier si le résultat de l'opération de justification est déjà mis en cache
                    resultats_cache_key = f"resultats_{cache_key}"
                    resultats_justification_cache = cache.get(resultats_cache_key)
                    if not resultats_justification_cache:
                        print("not found resultat in cache")
                        # Vérifier si le texte de l'URL est déjà mis en cache
                        texte_cache_key = f"text_{cache_key}"
                        texte_cache = cache.get(texte_cache_key)
                        if not texte_cache:
                            print("not found url in cache")
                            contenu_url = text_processor.get_html_content(url)
                            texte_html = text_processor.extract_text(contenu_url)
                            texte_cache = texte_html
                            # Mettre en cache le texte pour une durée déterminée, par exemple 1 heure
                            cache.set(texte_cache_key, texte_cache, timeout=3600)
                        resultats = justify_text(texte_cache, programme)
                        # Mettre en cache les résultats de justification
                        cache.set(resultats_cache_key, resultats, timeout=3600)
                    else:
                        resultats = resultats_justification_cache
                    resultats_justification.append(resultats)

                return JsonResponse({'message': resultats_justification})

        return JsonResponse({'erreur': 'Méthode non autorisée'}, status=405)

    def statistiques(request):
        fichiers = request.FILES.getlist('fichiers')
        urls = request.POST.getlist('urls')
        ngrame = request.POST.get('n')

        results = []

        # Create an instance of the TextProcessor class
        text_processor = TextProcessor()
        complete_text = ""
        # Traitement des fichiers
        for fichier in fichiers:
            contenu_fichier = fichier.read().decode("utf-8")
            # Vérifier si le texte du fichier est déjà mis en cache
            cache_key = f"text_{fichier.name}"
            texte_cache = cache.get(cache_key)
            if not texte_cache:
                print("not found cache")
                texte_cache = contenu_fichier
                # Mettre en cache le texte pour une durée déterminée, par exemple 1 heure
                cache.set(cache_key, texte_cache, timeout=3600)
            else:
                contenu_fichier = texte_cache
            # on concatene le contenu des fichiers pour en faire un long text
            complete_text += contenu_fichier

        # Traitement des URLs
        for url in urls:
            if url.strip():  # Vérifie si l'URL n'est pas vide après suppression des espaces
                # Vérifier si le texte de l'URL est déjà mis en cache
                cache_key = f"text_{url}"
                texte_cache = cache.get(cache_key)
                if not texte_cache:
                    print("not found cache")
                    contenu_url = text_processor.get_html_content(url)
                    texte_html = text_processor.extract_text(contenu_url)
                    texte_cache = texte_html
                    # Mettre en cache le texte pour une durée déterminée, par exemple 1 heure
                    cache.set(cache_key, texte_cache, timeout=3600)
                else:
                    texte_html = texte_cache
                # on concatene le contenu des url pour en faire un long text
                complete_text += texte_html
                
        # on calcule les stats sur ce "nouveau grand fichier"
        stats = text_processor.calculate_statistics(complete_text, int(ngrame))
        results.append(stats)

        return JsonResponse({'statistics': results})

    def generate(request):
        # Utilisez la variable de classe ngrames
        results = []
        first_word = request.POST.get('word')
        num_sentences = request.POST.get('num_sentences')
        if(num_sentences != ""):
            num = int(num_sentences)
        else:
            num = num_sentences

        text_processor = TextProcessor()

        resultat = text_processor.generation_text( first_word, num)
        results.append(resultat)
        
        return JsonResponse({'generate': results})