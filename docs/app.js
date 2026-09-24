


const species_path = "specieslist.csv"




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
                    if(row.display === '1'){
                        const newCard = document.createElement("div");
                        newCard.classList.add("frog-card")
                        const title = document.createElement("p");
                        title.textContent = row.scientific_name;
                        

                        let Audiopath = "docs/audio/" + row.scientific_name + ".mp3";
                        let Imagepath = "docs/img/" + row.scientific_name + ".PNG";
                        let Specpath = "docs/img/spec/" + row.scientific_name + ".png";
                        const audio = new Audio(Audiopath);
                        // audio.onerror = function(){
                        //     this.parentNode.style.display='none';};
                        audio.controls = true;

                        const img = document.createElement("img");
                        img.src = Imagepath

                        const spec = document.createElement("img")
                        spec.src = Specpath

                        
                        

                        newCard.appendChild(img)
                        newCard.appendChild(spec)
                        newCard.appendChild(title)
                        newCard.appendChild(audio)
                        
                        listElement.appendChild(newCard);
                    }    
                });
            }
        })
}



parseFile();
