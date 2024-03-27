import { CreatePredictDto } from "../../domain";
import { execSync } from "child_process";
import fs from 'fs';

export class PredictService {
    constructor() {}

    public async realizarPredict (createPredictDto: CreatePredictDto) {
        fs.writeFileSync('./data.json', JSON.stringify({
            "Glucosa": createPredictDto.glucosa,
            "Presion arterial": createPredictDto.presion_arterial,
            "Insulina": createPredictDto.insulina,
            "IMC": createPredictDto.imc,
            "DiabetesPedigríFunción": createPredictDto.diabetes_pedrigui_funcion,
        }));
        
        const result = execSync('python3 model.py');
        console.log(result.toString());
        
        return JSON.parse(
            fs.readFileSync('./data.json', { encoding: 'utf8' })
        );
    }
}