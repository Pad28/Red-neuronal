import { Validators } from "../../../config/validators";

export class CreatePredictDto {
    private constructor(
        public readonly edad: number,
        public readonly fumador_frecuente: number,
        public readonly cigarros_por_dia: number,
        public readonly medicamentos: number,
        public readonly diabetes: number,
        public readonly colesterol: number,
        public readonly sistolica: number,
        public readonly diastolica: number,
        public readonly imc: number,
        public readonly glucosa: number,
    ) {}

    static create(data: {[key: string]: any}): [string?, CreatePredictDto?] {
        try {
            const validators = new Validators(data);
            
            validators.requiredKeys(
                "edad",
                "fumador_frecuente",
                "cigarros_por_dia",
                "medicamentos",
                "diabetes",
                "colesterol",
                "sistolica",
                "diastolica",
                "imc",
                "glucosa",
            );

            validators.isNumber("edad");
            validators.isNumber("fumador_frecuente");
            validators.isNumber("cigarros_por_dia");
            validators.isNumber("medicamentos");
            validators.isNumber("diabetes");
            validators.isNumber("colesterol");
            validators.isNumber("sistolica");
            validators.isNumber("diastolica");
            validators.isFloat("imc");
            validators.isNumber("glucosa");


            return [undefined, new CreatePredictDto(
                data["edad"],
                data["fumador_frecuente"],
                data["cigarros_por_dia"],
                data["medicamentos"],
                data["diabetes"],
                data["colesterol"],
                data["sistolica"],
                data["diastolica"],
                data["imc"],
                data["glucosa"],
            )];
        } catch (error) {
            return [error as string];
        }
    }
}