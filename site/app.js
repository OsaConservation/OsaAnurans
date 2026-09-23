const species_path = "../specieslist.csv"

function parseFile(){
    // Fetch and parse the CSV file
        Papa.parse(species_path, {
            download: true,
            header: true, // Uses the first row as column headers
            complete: function(results) {
                const data = results.data.sort((a,b) => a.scientific_name.localeCompare(b.scientific_name));
                const listElement = document.getElementById("card-container");

                // Loop through each row and extract the 'scientific_name' column
                data.forEach(row => {
                    const newCard = document.createElement("div");
                    newCard.classList.add("frog-card")
                    const title = document.createElement("h2");
                    title.textContent = row.scientific_name;
                    newCard.appendChild(title)
                    listElement.appendChild(newCard);
                });
            }
        })
}


parseFile();
