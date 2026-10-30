const species_path = "../specieslist.csv"
const container = document.querySelector(".card-container")


function parseFile(){
    // Fetch and parse the CSV file
        Papa.parse(species_path, {
            delimiter: "\t",
            download: true,
            header: true, // Uses the first row as column headers
            complete: function(results) {
                // const data = results.data.sort((a,b) => a.scientific_name.localeCompare(b.scientific_name));
                const listElement = document.getElementById("card-container");

                // Loop through each row and extract the 'scientific_name' column
                results.data.forEach(row => {
                    if(row.display === '1'){
                        audio_files = Papa.parse(row.audio_files).data[0]
                        spec_files = Papa.parse(row.spec_files).data[0]
                        let newCard = createFrogCard(row.common_name, row.scientific_name, row.code, audio_files, spec_files);
                        
                        listElement.appendChild(newCard);
                    }    
                });
            }
        })
}


function createFrogCard(common_name, scientific_name, code, audio_files, spec_files){
    const newCard = document.createElement("div");
    newCard.classList.add("frog-card")
    const scientific = document.createElement("i");
    scientific.textContent = scientific_name;

    const common = document.createElement("p");
    common.textContent = common_name;

    const codeEl = document.createElement("p")
    codeEl.textContent = code;
    

    let Audiopath = "docs/static/" + scientific_name + "/audio/" + scientific_name + "0.mp3";
    let Imagepath = "docs/static/"+ scientific_name + "/img/"  + scientific_name + "0.webp";
    let Specpath = "docs/static/"+ scientific_name + "/img/spec/"  + scientific_name + "0.webp";

    // let Audiopath = "docs/audio/" + scientific_name + ".mp3";
    // let Imagepath = "docs/img/" + scientific_name + ".webp";
    // let Specpath = "docs/img/spec/" + scientific_name + ".webp";


    const audio = new Audio(Audiopath);
    // audio.onerror = function(){
    //     this.parentNode.style.display='none';};
    audio.controls = true;

    const img = document.createElement("img");
    img.src = Imagepath

    const spec = document.createElement("img")
    spec.src = Specpath;
    spec.classList.add("card-spec")

    const button = document.createElement("button");
    button.innerHTML = "See more"
    button.classList.add("popup-button");
    button.value = scientific_name

    
    const carousel = document.createElement("ul")



    var examples = audio_files.map((e, i) => [e, spec_files[i]])
    examples.forEach((file) =>{

        let Audiopath = "docs/static/" + scientific_name + "/audio/" + file[0]
        let audio = new Audio(Audiopath)
        audio.controls = true
        

        let Specpath = "docs/static/" + scientific_name + "/img/spec/" + file[1]

        let spec = document.createElement("img")
        spec.src = Specpath;
        spec.classList.add("spec")
        spec.classList.add("card-spec")

        let record = document.createElement("li")

        record.appendChild(spec)
        record.appendChild(audio)
        carousel.appendChild(record);

    })


    //newCard.appendChild(common);
    //newCard.appendChild(codeEl);
    newCard.appendChild(img);
    newCard.appendChild(carousel)
    // newCard.appendChild(spec);
    //newCard.appendChild(scientific);
    
    
    // newCard.appendChild(audio);
    newCard.appendChild(button);

    return newCard

}



const API_URL = "http://127.0.0.1:8080/submit";

document.getElementById("sendButton").addEventListener("click", async () => {

    const button = document.getElementById("sendButton");
    const status = document.getElementById("status");

    // Example data to send to Python
    const payload = {
        event: "button_clicked",
        timestamp: new Date().toISOString(),
        user_action: "manual_submission",
        value: 123
    };

    button.disabled = true;
    status.textContent = "Sending data...";

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const result = await response.json();

        console.log("Python response:", result);
        status.textContent = "Data successfully received by Python!";

    } catch (error) {
        console.error("Error:", error);
        status.textContent = "Failed to send data: " + error.message;

    } finally {
        button.disabled = false;
    }
});