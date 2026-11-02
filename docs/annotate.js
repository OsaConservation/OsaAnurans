const species_path = "../specieslist.csv"
// const container = document.querySelector(".card-container")
const resultsCarousel = document.querySelector(".results-carousel")
const listElement = document.getElementById("card-container");
const dropdown = document.getElementById('scientific')
const frogCard = document.getElementById('frog-card')



fake_data = {0: {'window_id': 25435,
  'score': -0.003451429307460785,
  'recording_id': 633,
  'offsets': [35.0, 40.0],
  'uri': 'sw2/1Jsw2_20261009_014000.wav',
  'dataset_name': 'grillos'},
 1: {'window_id': 23933,
  'score': -0.002235017716884613,
  'recording_id': 508,
  'offsets': [25.0, 30.0],
  'uri': 'sw1/1Jsw1_20260919_224000.wav',
  'dataset_name': 'grillos'},
 2: {'window_id': 23976,
  'score': -0.001905784010887146,
  'recording_id': 512,
  'offsets': [0.0, 5.0],
  'uri': 'sw1/1Jsw1_20260920_000000.wav',
  'dataset_name': 'grillos'},
 3: {'window_id': 24001,
  'score': -6.45071268081665e-05,
  'recording_id': 514,
  'offsets': [5.0, 10.0],
  'uri': 'sw1/1Jsw1_20260920_004000.wav',
  'dataset_name': 'grillos'},
 4: {'window_id': 22970,
  'score': -0.0010343044996261597,
  'recording_id': 428,
  'offsets': [10.0, 15.0],
  'uri': 'sw1/1Jsw1_20260917_225400.wav',
  'dataset_name': 'grillos'}}

const states = [
    { class: 'unlabeled', label: 'Unlabeled' },
    { class: 'positive', label: 'Positive' },
    { class: 'negative', label: 'Negative' },
    { class: 'uncertain', label: 'Uncertain' }
];

dropdown.addEventListener("change", function(){
    choice = this.value
    Papa.parse(species_path, {
            delimiter: "\t",
            download: true,
            header: true,
            complete: function(results){
                const index = results.data.findIndex(row => row.scientific_name === choice);
                row = results.data[index]
                
                audio_files = Papa.parse(row.audio_files).data[0]
                spec_files = Papa.parse(row.spec_files).data[0]
                let newCard = createFrogCard(row.common_name, row.scientific_name, row.code, audio_files, spec_files);
                frogCard.replaceChildren(frogCard.firstElementChild)
                frogCard.appendChild(newCard);
            }

        })
})

resultsCarousel.addEventListener("click", (e) =>{
    if(e.target.matches("button")){
        console.log(e.target)
        const key = e.target.value
        currentState = e.target.classList[0]

        var index = states.findIndex(row => row.class === currentState);

        index = (index + 1) % states.length;

        states.forEach(s => e.target.classList.remove(s.class))

        e.target.classList.add(states[index].class)
        e.target.textContent = states[index].label

    }
})


function parseFile(){
    // Fetch and parse the CSV file
        Papa.parse(species_path, {
            delimiter: "\t",
            download: true,
            header: true, // Uses the first row as column headers
            complete: function(results) {
                // const data = results.data.sort((a,b) => a.scientific_name.localeCompare(b.scientific_name));
                

                // Loop through each row and extract the 'scientific_name' column
                // results.data.forEach(row => {
                //     if(row.display === '1'){
                //         audio_files = Papa.parse(row.audio_files).data[0]
                //         spec_files = Papa.parse(row.spec_files).data[0]
                //         let newCard = createFrogCard(row.common_name, row.scientific_name, row.code, audio_files, spec_files);
                        
                //         listElement.appendChild(newCard);
                //     }    
                // });

                const data = []
                const all_names = []

                // Loop through each row and extract the 'scientific_name' column
                results.data.forEach(row => {
                    if(row.display === '1'){
                        data.push(row)
                        all_names.push(row.scientific_name)
                    }    
                });

                all_names.forEach(name => {
                    const option = document.createElement("option")
                    option.value = name
                    option.textContent = name
                    dropdown.appendChild(option)
                })

                choice = dropdown.value
                

                const index = data.findIndex(row => row.scientific_name === choice);

                row = data[index]
                
                audio_files = Papa.parse(row.audio_files).data[0]
                spec_files = Papa.parse(row.spec_files).data[0]
                let newCard = createFrogCard(row.common_name, row.scientific_name, row.code, audio_files, spec_files);
                
                frogCard.appendChild(newCard);
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
    

    let Audiopath = "static/" + scientific_name + "/audio/" + scientific_name + "0.mp3";
    let Imagepath = "static/"+ scientific_name + "/img/"  + scientific_name + "0.webp";
    let Specpath = "static/"+ scientific_name + "/img/spec/"  + scientific_name + "0.webp";

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

        let Audiopath = "static/" + scientific_name + "/audio/" + file[0]
        let audio = new Audio(Audiopath)
        audio.controls = true
        

        let Specpath = "static/" + scientific_name + "/img/spec/" + file[1]

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
    // newCard.appendChild(button);

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

function displayResults(data){

    for(const [key,value] of Object.entries(data)){
        let audio_path = "fake_data/download" + key + ".wav"
        let spec_path = "fake_data/output" + key + ".png"

        const audio = new Audio(audio_path);
        audio.controls = true;
        const spec = document.createElement("img")
        spec.src = spec_path;

        const button = document.createElement("button")
        button.classList.add("unlabeled")
        button.innerText = "Unlabeled"
        button.value = key

        let example = document.createElement("li")
        example.appendChild(spec)
        example.appendChild(audio)
        example.appendChild(button)
        resultsCarousel.appendChild(example)
        
        console.log(key,value)
    }
        

}

function createFrogCarousel(){
    Papa.parse(species_path, {
            delimiter: "\t",
            download: true,
            header: true,
            complete: function(results){
                const carousel = document.querySelector(".frog-carousel")

                results.data.forEach(row => {
                    if(row.display === '1'){
                        audio_files = Papa.parse(row.audio_files).data[0]
                        spec_files = Papa.parse(row.spec_files).data[0]
                        let newCard = createFrogCard(row.common_name, row.scientific_name, row.code, audio_files, spec_files);
                        let newLi = document.createElement("li")

                        newLi.appendChild(newCard)
                        carousel.appendChild(newLi);
                    }
                    
                })
               
                
                
            }

        })
}



createFrogCarousel()
displayResults(fake_data)
parseFile()
// Recieving

    // window_id:
    // recording_id:
    // offsets:
    // dataset_name:
    // uri:
    // score:
    // labels:

// Sending

    // id: (annotation id = -1)
    // recording_id:
    // offsets:
    // label:
    // label_type:
    // provenance:
