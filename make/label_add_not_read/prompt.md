Analyse le contenu de cet email et suggère le label le plus pertinent parmi la liste suivante :

- **ILLIDAN** : Si l'email mentionne mon fils à l'école ISG Bordeaux, dont le prénom est Illidan.
- **MAISON** : Si l'email concerne notre appartement au 35 chemin du Chapitre, résidence Les Charmilles.
- **VOYAGES** : Si l'email provient d'une société impliquée dans les voyages.
- **SOCIAL** : Si l'email contient des éléments de divertissement tels que des contenus humoristiques, des recommandations de loisirs, des jeux, ou des liens vers des vidéos et événements amusants.
- **DIVERTISSEMENT** : Si l'email contient des éléments de divertissement, similaires à ceux mentionnés pour le label "SOCIAL".
- **PUB** : Si l'email est publicitaire, avec des promotions, des appels à l'action, un ton commercial, ou un expéditeur d'entreprise.
- **ADMIN** : Si l'email est lié à des sujets administratifs, tels que les impôts, la sécurité sociale, ou les documents officiels. Exemples de mots-clés : "impôts", "sécurité sociale", "déclaration fiscale", "URSSAF", "DGFiP", "CAF", "CNAV".
- **SPAM** : Si l'email présente des éléments suspects tels que des fautes d'orthographe, des demandes urgentes d'informations personnelles, des liens ou pièces jointes inhabituels, ou une adresse d'expéditeur douteuse, suggérant qu'il s'agit de spam, phishing, ou escroquerie.
- **FINANCE** : Si l'email concerne des achats ou des transactions financières. Exemples de mots-clés : "facture", "reçu", "confirmation de commande", "paiement", "numéro de transaction".
- **AGENDA** : Si l'email contient des notifications de rendez-vous. Exemples de mots-clés : "rendez-vous", "meeting", "réunion", "consultation", "rappel de rendez-vous", "agenda".
- **AUTRE** : Si l'email ne correspond à aucun autre label.

- **Contenu de l'email** : "{{11.text}}"
- **Expéditeur** : "{{1.from.address}}"
- **Objet** : "{{1.subject}}"

- Réponds au format JSON avec un seul objet contenant l'attribut `label`.