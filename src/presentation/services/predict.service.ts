import { CreatePredictDto } from "../../domain";
import { execSync } from "child_process";
import fs from 'fs';

export class PredictService {
    constructor() {}

    public async realizarPredict (createPredictDto: CreatePredictDto) {
        
        fs.writeFileSync('./data.json', JSON.stringify({
            "Edad": createPredictDto.edad,
            "Fumador frecuente": createPredictDto.fumador_frecuente,
            "Cigarros por día": createPredictDto.cigarros_por_dia,
            "Uso de medicamentos para la presión arterial": createPredictDto.medicamentos,
            "Diabetes": createPredictDto.medicamentos,
            "Nivel de colesterol": createPredictDto.colesterol,
            "Presión arterial sistólica": createPredictDto.sistolica,
            "Presión arterial diastólica": createPredictDto.diastolica,
            "IMC": createPredictDto.imc,
            "Glucosa": createPredictDto.glucosa,
        }));
        
        const result = execSync('python3 model.py');
        console.log(result.toString());
        
        return JSON.parse(
            fs.readFileSync('./data.json', { encoding: 'utf8' })
        );
    }
}